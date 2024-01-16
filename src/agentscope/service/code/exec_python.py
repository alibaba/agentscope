# -*- coding: utf-8 -*-
""" Execute python code functions. """
import builtins
import contextlib
import inspect
import io
import multiprocessing
import os
import platform
import re
import resource
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

from agentscope.utils.common import create_tempdir, timer
from agentscope.constants import ServiceExecStatus
from agentscope.service.service_response import ServiceResponse


def execute_python_code(
    code: str = "",
    timeout: Optional[Union[int, float]] = 300,
    use_docker: Optional[Union[bool, str]] = None,
    maximum_memory_bytes: Optional[int] = None,
) -> ServiceResponse:
    """
    Execute a string of Python code, optionally inside a Docker container.

    This function can run Python code provided in string format. It has the
    option to execute the code within a Docker container to provide an
    additional layer of security, especially important when running
    untrusted code.

    WARNING: If `use_docker` is set to `False`, the code will be run
    directly in the host system's environment. This poses a potential
    security risk if the code is untrusted. Only disable Docker if you are
    confident in the safety of the code being executed.

    Args:
        code (`str`, optional):
            The Python code to execute, provided as a string. Default is an
            empty string.

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

    @timer(timeout)
    def sys_execute() -> None:
        is_success = False
        with create_tempdir():
            # These system calls are needed when cleaning up tempdir.
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            sys_python_guard(maximum_memory_bytes)
            output_buffer, error_buffer = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(
                output_buffer,
            ), contextlib.redirect_stderr(error_buffer):
                try:
                    exec(code)
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

    manager = multiprocessing.Manager()
    shared_list = manager.list()

    p = multiprocessing.Process(target=sys_execute)
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
    # TODO: delete it or make it configurable when release.
    # sources pip install from
    pypi_mirror = "http://pypi.douban.com/simple"
    pypi_trusted_host = "pypi.douban.com"

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
            "import contextlib, signal\nfrom typing import Any, Generator, "
            "Optional, Union\n"
            + timer_code
            + f"\nwith timer({timeout}):\n    "
        )

        # Construct the command to be executed inside the timer context
        exec_code_with_timer = f"""{exec_code_with_timer}
            exec('''{exec_code}''')
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
                    f" {pypi_mirror} "
                    f"--trusted-host {pypi_trusted_host}"
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
    run_args = {"image": image_name, "detach": True, "network_disabled": False}
    if maximum_memory_bytes is not None:
        run_args["mem_limit"] = maximum_memory_bytes

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
