# -*- coding: utf-8 -*-
"""The init function for the package."""
import json
import os
from typing import Optional, Union, Sequence
from agentscope import agents
from .agents import AgentBase
from ._runtime import _runtime
from .logging import LOG_LEVEL, setup_logger
from .manager import FileManager, MonitorManager
from .manager import ModelManager
from .constants import _DEFAULT_SAVE_DIR
from .constants import _DEFAULT_LOG_LEVEL
from .constants import _DEFAULT_CACHE_DIR
from .studio._client import _studio_client

# init setting
_INIT_SETTINGS = {}

# init the singleton class by default settings to avoid reinit in subprocess
# especially in spawn mode, which will copy the object from the parent process
# to the child process rather than re-import the module (fork mode)
FileManager()
ModelManager()
MonitorManager()


def init(
    model_configs: Optional[Union[dict, str, list]] = None,
    project: Optional[str] = None,
    name: Optional[str] = None,
    disable_saving: bool = False,
    save_dir: str = _DEFAULT_SAVE_DIR,
    save_log: bool = True,
    save_code: bool = True,
    save_api_invoke: bool = False,
    cache_dir: str = _DEFAULT_CACHE_DIR,
    use_monitor: bool = True,
    logger_level: LOG_LEVEL = _DEFAULT_LOG_LEVEL,
    runtime_id: Optional[str] = None,
    agent_configs: Optional[Union[str, list, dict]] = None,
    studio_url: Optional[str] = None,
) -> Sequence[AgentBase]:
    """A unified entry to initialize the package, including model configs,
    runtime names, saving directories and logging settings.

    Args:
        model_configs (`Optional[Union[dict, str, list]]`, defaults to `None`):
            A dict, a list of dicts, or a path to a json file containing
            model configs.
        project (`Optional[str]`, defaults to `None`):
            The project name, which is used to identify the project.
        name (`Optional[str]`, defaults to `None`):
            The name for runtime, which is used to identify this runtime.
        disable_saving (`bool`, defaults to `False`):
            Whether to disable saving files. If `True`, this will override
            the `save_log`, `save_code`, and `save_api_invoke` parameters.
        runtime_id (`Optional[str]`, defaults to `None`):
            The id for runtime, which is used to identify this runtime. Use
            `None` will generate a random id.
        save_dir (`str`, defaults to `./runs`):
            The directory to save logs, files, codes, and api invocations.
            If `dir` is `None`, when saving logs, files, codes, and api
            invocations, the default directory `./runs` will be created.
        save_log (`bool`, defaults to `False`):
            Whether to save logs locally.
        save_code (`bool`, defaults to `False`):
            Whether to save codes locally.
        save_api_invoke (`bool`, defaults to `False`):
            Whether to save api invocations locally, including model and web
            search invocation.
        cache_dir (`str`):
            The directory to cache files. In Linux/Mac, the dir defaults to
        `~/.cache/agentscope`. In Windows, the dir defaults to
        `C:\\users\\<username>\\.cache\\agentscope`.
        use_monitor (`bool`, defaults to `True`):
            Whether to activate the monitor.
        logger_level (`LOG_LEVEL`, defaults to `"INFO"`):
            The logging level of logger.
        agent_configs (`Optional[Union[str, list, dict]]`, defaults to `None`):
            The config dict(s) of agents or the path to the config file,
            which can be loaded by json.loads(). One agent config should
            cover the required arguments to initialize a specific agent
            object, otherwise the default values will be used.
        studio_url (`Optional[str]`, defaults to `None`):
            The url of the agentscope studio.
    """
    init_process(
        model_configs=model_configs,
        project=project,
        name=name,
        runtime_id=runtime_id,
        disable_saving=disable_saving,
        save_dir=save_dir,
        save_log=save_log,
        save_code=save_code,
        save_api_invoke=save_api_invoke,
        cache_dir=cache_dir,
        use_monitor=use_monitor,
        logger_level=logger_level,
        studio_url=studio_url,
    )

    # save init settings for subprocess
    _INIT_SETTINGS["model_configs"] = model_configs
    _INIT_SETTINGS["project"] = _runtime.project
    _INIT_SETTINGS["name"] = _runtime.name
    _INIT_SETTINGS["runtime_id"] = _runtime.runtime_id
    _INIT_SETTINGS["disable_saving"] = disable_saving
    _INIT_SETTINGS["save_dir"] = save_dir
    _INIT_SETTINGS["save_code"] = False
    _INIT_SETTINGS["save_api_invoke"] = save_api_invoke
    _INIT_SETTINGS["save_log"] = save_log
    _INIT_SETTINGS["logger_level"] = logger_level
    _INIT_SETTINGS["use_monitor"] = use_monitor
    _INIT_SETTINGS["cache_dir"] = cache_dir

    # Load config and init agent by configs
    if agent_configs is not None:
        if isinstance(agent_configs, str):
            with open(agent_configs, "r", encoding="utf-8") as file:
                configs = json.load(file)
        elif isinstance(agent_configs, dict):
            configs = [agent_configs]
        else:
            configs = agent_configs

        # setup agents
        agent_objs = []
        for config in configs:
            agent_cls = getattr(agents, config["class"])
            agent_args = config["args"]
            agent = agent_cls(**agent_args)
            agent_objs.append(agent)
        return agent_objs
    return []


def init_process(
    model_configs: Optional[Union[dict, str, list]],
    project: Optional[str],
    name: Optional[str],
    runtime_id: Optional[str],
    disable_saving: bool,
    save_dir: str,
    save_api_invoke: bool,
    save_log: bool,
    save_code: bool,
    cache_dir: str,
    use_monitor: bool,
    logger_level: LOG_LEVEL,
    studio_url: Optional[str] = None,
) -> None:
    """An entry to initialize the package in a process.

    Args:
        project (`Optional[str]`, defaults to `None`):
            The project name, which is used to identify the project.
        name (`Optional[str]`, defaults to `None`):
            The name for runtime, which is used to identify this runtime.
        runtime_id (`Optional[str]`, defaults to `None`):
            The id for runtime, which is used to identify this runtime.
        disable_saving (`bool`):
            Whether to disable saving files. If `True`, this will override
            the `save_log`, `save_code`, and `save_api_invoke` parameters.
        save_dir (`str`, defaults to `./runs`):
            The directory to save logs, files, codes, and api invocations.
            If `dir` is `None`, when saving logs, files, codes, and api
            invocations, the default directory `./runs` will be created.
        save_api_invoke (`bool`, defaults to `False`):
            Whether to save api invocations locally, including model and web
            search invocation.
        model_configs (`Optional[Sequence]`, defaults to `None`):
            A sequence of pre-init model configs.
        save_log (`bool`, defaults to `False`):
            Whether to save logs locally.
        save_code (`bool`):
            Whether to save codes locally.
        cache_dir (`str`):
            The directory to cache files. In Linux/Mac, the dir defaults to
            `~/.cache/agentscope`. In Windows, the dir defaults to
            `C:\\users\\<username>\\.cache\\agentscope`.
        use_monitor (`bool`, defaults to `True`):
            Whether to activate the monitor.
        logger_level (`LOG_LEVEL`, defaults to `"INFO"`):
            The logging level of logger.
        studio_url (`Optional[str]`, defaults to `None`):
            The url of the agentscope studio.
    """
    # Init the runtime
    if project is not None:
        _runtime.project = project

    if name is not None:
        _runtime.name = name

    if runtime_id is not None:
        _runtime.runtime_id = runtime_id

    # Init file manager
    file_manager = FileManager.get_instance()
    file_manager.initialize(
        disable_saving,
        save_dir,
        save_log,
        save_code,
        save_api_invoke,
        cache_dir,
    )

    # Init logger
    setup_logger(file_manager.run_dir, logger_level)

    # Init model manager
    if model_configs is not None:
        ModelManager.get_instance().load_model_configs(model_configs)

    # Init monitor
    MonitorManager.get_instance().initialize(
        not disable_saving and use_monitor,
    )

    # Init studio client, which will push messages to web ui and fetch user
    # inputs from web ui
    if studio_url is not None:
        _studio_client.initialize(_runtime.runtime_id, studio_url)
        # Register in AgentScope Studio
        _studio_client.register_running_instance(
            project=_runtime.project,
            name=_runtime.name,
            timestamp=_runtime.timestamp,
            run_dir=file_manager.run_dir,
            pid=os.getpid(),
        )
