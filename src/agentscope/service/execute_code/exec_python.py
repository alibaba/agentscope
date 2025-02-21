# -*- coding: utf-8 -*-
"""Service to execute python code."""
import builtins
import contextlib
import inspect
import io
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from hashlib import md5
from typing import Optional, Union, Tuple

from loguru import logger

try:
    import docker
    from docker.errors import APIError, ImageNotFound
except ImportError:
    docker = None
try:
    import resource
except (ModuleNotFoundError, ImportError):
    resource = None

from ...utils.common import create_tempdir, timer
from ..service_status import ServiceExecStatus
from ..service_response import ServiceResponse
from ...constants import (
    _DEFAULT_PYPI_MIRROR,
    _DEFAULT_TRUSTED_HOST,
)


def execute_python_code(
    code: str,
    timeout: Optional[Union[int, float]] = 300,
    use_docker: Optional[Union[bool, str]] = None,
    maximum_memory_bytes: Optional[int] = None,
) -> ServiceResponse:
    """
    Execute a piece of python code.

    This function can run Python code provided in string format. It has the
    option to execute the code within a Docker container to provide an
    additional layer of security, especially important when running
    untrusted code.

    WARNING: If `use_docker` is set to `False`, the `code` will be run
    directly in the host system's environment. This poses a potential
    security risk if the code is untrusted. Only disable Docker if you are
    confident in the safety of the code being executed.

    Args:
        code (`str`, optional):
            The Python code to be executed.

        timeout (`Optional[Union[int, float]]`, defaults to `300`):
            The maximum time (in seconds) allowed for the code to run. If
            the code execution time exceeds this limit, it will be
            terminated. Set to `None` for no time limit. Default is 300.

        use_docker (`Optional[Union[bool, str]]`, defaults to `None`):
            Determines whether to execute the code within a Docker
            container. If `False`, the system's native Python environment is
            used. When set to `None`, the function checks for Docker's
            availability and uses it if present. When set to some string,
            will use the docker with string as the image name. Default is
            `None`.

        maximum_memory_bytes (`Optional[int]`, defaults to `None`):
            The memory limit in bytes for the code execution. If not
            specified, there is no memory limit imposed.

    Returns:
        `ServiceResponse`: A ServiceResponse containing two elements:
        `output` and `error`. Both `output` and `error` are strings that
        capture the standard output and standard error of the code
        execution, respectively.

    Note:
        IPython-specific operations such as `plt.show()` for displaying
        matplotlib plots are currently not supported. This limitation stems
        from the non-interactive nature of the execution environment.

        The argument `timeout` is not available in Windows OS, since the
        since `signal.setitimer` is only available in Unix.

    """
    # Check if the `use_docker` flag has been explicitly set by the user.
    if use_docker is None:
        # If `use_docker` is not set, determine whether to use Docker based on
        # the availability of the Docker module in the environment.
        if docker is None:
            # If the Docker module is not available, default to not using
            # Docker.
            use_docker = False
        else:
            # If the Docker module is available, default to using Docker.
            use_docker = True

    if use_docker:
        response = _execute_python_code_docker(
            code,
            timeout,
            use_docker,
            maximum_memory_bytes,
        )
    else:
        response = _execute_python_code_sys(
            code,
            timeout,
            maximum_memory_bytes,
        )

    return response


def _sys_execute(
    code: str,
    shared_list: list,
    maximum_memory_bytes: int,
    timeout: int,
) -> None:
    """
    Executes the given Python code in a controlled environment, capturing
    the output and errors.

    Parameters:
        code (str): The Python code to be executed.
        shared_list (ListProxy): A list proxy managed by a
            multiprocessing.Manager to which the output and error messages
            will be appended, along with a success flag.
        maximum_memory_bytes (int): The maximum amount of memory in bytes
            that the execution is allowed to use.
        timeout (int): The maximum amount of time in seconds that the code
            is allowed to run.

    Returns:
        None: This function does not return anything. It appends the results
            to the shared_list.
    """
    is_success = False
    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        sys_python_guard(maximum_memory_bytes)
        output_buffer, error_buffer = io.StringIO(), io.StringIO()
        exec_namespace = {}  # Create a shared namespace
        with timer(timeout), contextlib.redirect_stdout(
            output_buffer,
        ), contextlib.redirect_stderr(error_buffer):
            try:
                exec(code, exec_namespace, exec_namespace)  # Use the namespace
                is_success = True
            except Exception:
                error_buffer.write(traceback.format_exc())

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
    shared_list.extend(
        [output_buffer.getvalue(), error_buffer.getvalue(), is_success],
    )


def _execute_python_code_sys(
    code: str = "",
    timeout: Optional[Union[int, float]] = None,
    maximum_memory_bytes: Optional[int] = None,
) -> ServiceResponse:
    """
    Execute string of python code in system environments.

    WARNING: This function is designed to execute code generated by models
    that have not been explicitly trusted. The likelihood of such code
    being maliciously harmful is low, yet there exists a risk of unintended
    destructive behavior arising from the model's limitations or misalignment.
    """
    logger.warning(
        "Executing code in system environments. There exists a risk of "
        "unintended destructive behavior. Please consider using a "
        "containerized environment.",
    )

    manager = multiprocessing.Manager()
    shared_list = manager.list()

    p = multiprocessing.Process(
        target=_sys_execute,
        args=(
            code,
            shared_list,
            maximum_memory_bytes,
            timeout,
        ),
    )
    p.start()
    p.join()
    if p.is_alive():
        p.kill()
    output, error, status = shared_list[0], shared_list[1], shared_list[2]
    if status:
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content=output,
        )
    else:
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content=f"{output}\n{error}",
        )


def _execute_python_code_docker(
    code: str = "",
    timeout: Optional[Union[int, float]] = None,
    use_docker: Optional[Union[bool, str]] = True,
    maximum_memory_bytes: Optional[int] = None,
) -> ServiceResponse:
    """
    Execute string of python code in containerized environments.

    If ImportErrors occur, this function will attempt to install the missing
    packages and retry execution until no ImportErrors are found or until
    execution succeeds.
    """

    def docker_execute(
        exec_code: str,
        max_retries: int = 5,
    ) -> Tuple:
        """Helper function to execute code inside the container."""
        missing_modules = []
        # Extract source code with wrapper timer
        timer_code = str(inspect.getsource(timer))
        is_success = False

        # Construct the timer context manager code
        exec_code_with_timer = (
            "import contextlib, signal\n"
            "from typing import Any, Generator, Optional, Union\n"
            + timer_code
            + f"\nwith timer({timeout}):\n"
        )

        # Add indentation to the exec_code
        indented_exec_code = "\n".join(
            "    " + line for line in exec_code.splitlines()
        )

        # Construct the complete code to execute
        exec_code_with_timer = f"""{exec_code_with_timer}{indented_exec_code}
"""

        # Create a temporary file to store the commands to run
        code_hash = md5(code.encode()).hexdigest()
        file_name = f"tmp_code_{code_hash}.py"
        with open(file_name, "w", encoding="utf-8") as exec_code_file:
            exec_code_file.write(exec_code_with_timer)

        try:
            for _ in range(max_retries):
                # Check if there are missing modules to install
                install_command = (
                    f"pip install -q {' '.join(missing_modules)} -i"
                    f" {_DEFAULT_PYPI_MIRROR} "
                    f"--trusted-host {_DEFAULT_TRUSTED_HOST}"
                    if missing_modules
                    else ""
                )

                # Construct the Docker command
                docker_command = (
                    f"{install_command} && python /app/{file_name}"
                )
                docker_command = docker_command.strip("& ")

                container = client.containers.run(
                    image=image_name,
                    command=docker_command,
                    volumes={os.getcwd(): {"bind": "/app", "mode": "rw"}},
                    working_dir="/app",
                    detach=True,
                    network_disabled=False,
                    mem_limit=maximum_memory_bytes,
                )
                wait_response = container.wait()
                docker_out = container.logs(stdout=True, stderr=False).decode(
                    "utf-8",
                )
                docker_err = container.logs(stdout=False, stderr=True).decode(
                    "utf-8",
                )
                is_success = wait_response.get("StatusCode", None) == 0
                # Check for ImportError or ModuleNotFoundError in stderr
                if (
                    "ImportError" not in docker_err
                    and "ModuleNotFoundError" not in docker_err
                ):
                    break

                # Extract the name of the missing module
                missing_module_match = re.search(
                    r"No module named '(\w+)'",
                    docker_err,
                )
                if missing_module_match:
                    missing_modules.append(missing_module_match.group(1))
                else:
                    # If a missing module cannot be determined, do not retry
                    break
        except Exception as e:
            logger.error(e)
            return "", str(e), False
        finally:
            # Clean up the temporary file
            if os.path.exists(file_name):
                os.remove(file_name)

        return docker_out, docker_err, is_success

    client = docker.from_env()  # Initialize Docker client

    # Step 1. Pull images & enter images
    image_name = "python:3.9.12" if use_docker is True else use_docker

    # Check if the image exists locally before pulling
    local_images = [
        tag for image in client.images.list() for tag in image.tags
    ]
    if image_name not in local_images:
        try:
            # Pull the image if it does not exist locally
            client.images.pull(image_name)
        except (ImageNotFound, APIError) as e:
            return ServiceResponse(
                status=ServiceExecStatus.ERROR,
                content=f"Failed to pull Docker image: {e}",
            )

    # Step 2. Execute code and catch Import Error and re-install
    # Try to execute the code and retry if ImportErrors are encountered
    output, error, status = docker_execute(code)

    if status:
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content=output,
        )
    else:
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content=f"{output}\n{error}",
        )


def sys_python_guard(maximum_memory_bytes: Optional[int] = None) -> None:
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    The implementation of this function are modified from
    https://github.com/openai/human-eval/blob/master/human_eval/execution.py
    """

    if resource is not None:
        if maximum_memory_bytes is not None:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (maximum_memory_bytes, maximum_memory_bytes),
            )
            resource.setrlimit(
                resource.RLIMIT_DATA,
                (maximum_memory_bytes, maximum_memory_bytes),
            )
            if not platform.uname().system == "Darwin":
                resource.setrlimit(
                    resource.RLIMIT_STACK,
                    (maximum_memory_bytes, maximum_memory_bytes),
                )

    # Disable builtins functions
    builtins_funcs_to_disable = ["exit", "quit"]
    for func_name in builtins_funcs_to_disable:
        setattr(builtins, func_name, None)

    # Disable os functions
    os.environ["OMP_NUM_THREADS"] = "1"
    os_funcs_to_disable = [
        "kill",
        "system",
        "putenv",
        "remove",
        "removedirs",
        "rmdir",
        "fchdir",
        "setuid",
        "fork",
        "forkpty",
        "killpg",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "fchmod",
        "fchown",
        "chmod",
        "chown",
        "chroot",
        "lchflags",
        "lchmod",
        "lchown",
        "getcwd",
        "chdir",
    ]
    for func_name in os_funcs_to_disable:
        setattr(os, func_name, None)

    # Disable shutil functions
    shutil_funcs_to_disable = ["rmtree", "move", "chown"]
    for func_name in shutil_funcs_to_disable:
        setattr(shutil, func_name, None)

    # Disable subprocess functions
    subprocess_funcs_to_disable = ["Popen"]
    for func_name in subprocess_funcs_to_disable:
        setattr(subprocess, func_name, None)

    __builtins__["help"] = None

    # Disable sys modules
    sys_modules_to_disable = [
        "ipdb",
        "joblib",
        "resource",
        "psutil",
        "tkinter",
    ]
    for module_name in sys_modules_to_disable:
        sys.modules[module_name] = None
