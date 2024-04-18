# -*- coding: utf-8 -*-
""" Base class for Agent """

from __future__ import annotations
from abc import ABCMeta
from typing import Optional
from typing import Sequence
from typing import Union
from typing import Any
from typing import Type
import uuid
from loguru import logger

from agentscope.agents.operator import Operator
from agentscope.models import load_model_by_config_name
from agentscope.memory import TemporaryMemory


class _AgentMeta(ABCMeta):
    """The meta-class for agent.

    1. record the init args into `_init_settings` field.
    2. register class name into `registry` field.
    """

    def __init__(cls, name: Any, bases: Any, attrs: Any) -> None:
        if not hasattr(cls, "registry"):
            cls.registry = {}
        else:
            if name in cls.registry:
                logger.warning(
                    f"Agent class with name [{name}] already exists.",
                )
            else:
                cls.registry[name] = cls
        super().__init__(name, bases, attrs)

    def __call__(cls, *args: tuple, **kwargs: dict) -> Any:
        if "to_dist" in kwargs and kwargs["to_dist"] is True:
            from .rpc_agent import RpcAgent

            if len(args) > 0:
                logger.warning(
                    "Please only use keywords arguments to init your agents."
                    "\ne.g.\nagent = YourAgentClass(name='agent_name', ...)",
                    "",
                )
            kwargs.pop("to_dist")
            if cls is not RpcAgent and not issubclass(cls, RpcAgent):
                return RpcAgent(
                    name=kwargs["name"],  # type: ignore[arg-type]
                    host=kwargs.pop(  # type: ignore[arg-type]
                        "host",
                        "localhost",
                    ),
                    port=kwargs.pop("port", None),  # type: ignore[arg-type]
                    max_pool_size=kwargs.pop(  # type: ignore[arg-type]
                        "max_pool_size",
                        8192,
                    ),
                    max_timeout_seconds=kwargs.pop(  # type: ignore[arg-type]
                        "max_timeout_seconds",
                        1800,
                    ),
                    local_mode=kwargs.pop(  # type: ignore[arg-type]
                        "local_mode",
                        True,
                    ),
                    lazy_launch=kwargs.pop(  # type: ignore[arg-type]
                        "lazy_launch",
                        True,
                    ),
                    agent_id=kwargs.pop(  # type: ignore[arg-type]
                        "agent_id",
                        cls.generate_agent_id(),
                    ),
                    connect_existing=kwargs.pop(  # type: ignore[arg-type]
                        "connect_existing",
                        False,
                    ),
                    agent_class=cls,
                    agent_configs={
                        "args": args,
                        "kwargs": kwargs,
                        "class_name": cls.__name__,
                    },
                )
        instance = super().__call__(*args, **kwargs)
        instance._init_settings = {
            "args": args,
            "kwargs": kwargs,
            "class_name": cls.__name__,
        }
        return instance


# todo: add a unique agent_type field to distinguish different agent class
def get_agent_class(agent_class_name: str) -> Type[AgentBase]:
    """Get the agent class based on the specific agent class name.

    Args:
        agent_class_name (`str`): the name of the agent class.

    Raises:
        ValueError: Agent class name not exits.

    Returns:
        Type[AgentBase]: the AgentBase sub-class.
    """
    if agent_class_name not in AgentBase.registry:
        raise ValueError()
    return AgentBase.registry[agent_class_name]


class AgentBase(Operator, metaclass=_AgentMeta):
    """Base class for all agents.

    All agents should inherit from this class and implement the `reply`
    function.
    """

    _version: int = 1

    def __init__(
        self,
        name: str,
        sys_prompt: Optional[str] = None,
        model_config_name: str = None,
        use_memory: bool = True,
        memory_config: Optional[dict] = None,
    ) -> None:
        r"""Initialize an agent from the given arguments.

        Args:
            name (`str`):
                The name of the agent.
            sys_prompt (`Optional[str]`):
                The system prompt of the agent, which can be passed by args
                or hard-coded in the agent.
            model_config_name (`str`, defaults to None):
                The name of the model config, which is used to load model from
                configuration.
            use_memory (`bool`, defaults to `True`):
                Whether the agent has memory.
            memory_config (`Optional[dict]`):
                The config of memory.
        """
        self.name = name
        self.memory_config = memory_config

        if sys_prompt is not None:
            self.sys_prompt = sys_prompt

        # TODO: support to receive a ModelWrapper instance
        if model_config_name is not None:
            self.model = load_model_by_config_name(model_config_name)

        if use_memory:
            self.memory = TemporaryMemory(memory_config)
        else:
            self.memory = None

        # The global unique id of this agent
        self._agent_id = self.__class__.generate_agent_id()

        # The audience of this agent, which means if this agent generates a
        # response, it will be passed to all agents in the audience.
        self._audience = None

    @classmethod
    def generate_agent_id(cls) -> str:
        """Generate the agent_id of this agent instance"""
        # TODO: change cls.__name__ into a global unique agent_type
        return f"{cls.__name__}_{uuid.uuid4().hex}"

    def reply(self, x: dict = None) -> dict:
        """Define the actions taken by this agent.

        Args:
            x (`dict`, defaults to `None`):
                Dialog history and some environment information

        Returns:
            The agent's response to the input.

        Note:
            Given that some agents are in an adversarial environment,
            their input doesn't include the thoughts of other agents.
        """
        raise NotImplementedError(
            f"Agent [{type(self).__name__}] is missing the required "
            f'"reply" function.',
        )

    def load_from_config(self, config: dict) -> None:
        """Load configuration for this agent.

        Args:
            config (`dict`): model configuration
        """

    def export_config(self) -> dict:
        """Return configuration of this agent.

        Returns:
            The configuration of current agent.
        """
        return {}

    def load_memory(self, memory: Sequence[dict]) -> None:
        r"""Load input memory."""

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """Calling the reply function, and broadcast the generated
        response to all audiences if needed."""
        res = self.reply(*args, **kwargs)

        # broadcast to audiences if needed
        if self._audience is not None:
            self._broadcast_to_audience(res)

        return res

    def speak(
        self,
        content: Union[str, dict],
    ) -> None:
        """Speak out the content generated by the agent."""
        logger.chat(content)

    def observe(self, x: Union[dict, Sequence[dict]]) -> None:
        """Observe the input, store it in memory without response to it.

        Args:
            x (`Union[dict, Sequence[dict]]`):
                The input message to be recorded in memory.
        """
        if self.memory:
            self.memory.add(x)

    def reset_audience(self, audience: Sequence[AgentBase]) -> None:
        """Set the audience of this agent, which means if this agent
        generates a response, it will be passed to all audiences.

        Args:
            audience (`Sequence[AgentBase]`):
                The audience of this agent, which will be notified when this
                agent generates a response message.
        """
        # TODO: we leave the consideration of nested msghub for future.
        #  for now we suppose one agent can only be in one msghub
        self._audience = [_ for _ in audience if _ != self]

    def clear_audience(self) -> None:
        """Remove the audience of this agent."""
        # TODO: we leave the consideration of nested msghub for future.
        #  for now we suppose one agent can only be in one msghub
        self._audience = None

    def rm_audience(
        self,
        audience: Union[Sequence[AgentBase], AgentBase],
    ) -> None:
        """Remove the given audience from the Sequence"""
        if not isinstance(audience, Sequence):
            audience = [audience]

        for agent in audience:
            if self._audience is not None and agent in self._audience:
                self._audience.pop(self._audience.index(agent))
            else:
                logger.warning(
                    f"Skip removing agent [{agent.name}] from the "
                    f"audience for its inexistence.",
                )

    def _broadcast_to_audience(self, x: dict) -> None:
        """Broadcast the input to all audiences."""
        for agent in self._audience:
            agent.observe(x)

    @property
    def agent_id(self) -> str:
        """The unique id of this agent.

        Returns:
            str: agent_id
        """
        return self._agent_id

    def to_dist(
        self,
        host: str = "localhost",
        port: int = None,
        max_pool_size: int = 8192,
        max_timeout_seconds: int = 1800,
        local_mode: bool = True,
        lazy_launch: bool = True,
        launch_server: bool = None,
    ) -> AgentBase:
        """Convert current agent instance into a distributed version.

        Args:
            host (`str`, defaults to `"localhost"`):
                Hostname of the rpc agent server.
            port (`int`, defaults to `None`):
                Port of the rpc agent server.
            max_pool_size (`int`, defaults to `8192`):
                Max number of task results that the server can accommodate.
            max_timeout_seconds (`int`, defaults to `1800`):
                Timeout for task results.
            local_mode (`bool`, defaults to `True`):
                Whether the started rpc server only listens to local
                requests.
            lazy_launch (`bool`, defaults to `True`):
                Only launch the server when the agent is called.
            launch_server(`bool`, defaults to `None`):
                This field has been deprecated and will be removed in
                future releases.

        Returns:
            `AgentBase`: the wrapped agent instance with distributed
            functionality
        """
        from .rpc_agent import RpcAgent

        if issubclass(self.__class__, RpcAgent):
            return self
        if launch_server is not None:
            logger.warning(
                "`launch_server` has been deprecated and will be removed in "
                "future releases. When `host` and `port` is not provided, the "
                "agent server will be launched automatically.",
            )
        return RpcAgent(
            name=self.name,
            agent_class=self.__class__,
            agent_configs=self._init_settings,
            host=host,
            port=port,
            max_pool_size=max_pool_size,
            max_timeout_seconds=max_timeout_seconds,
            local_mode=local_mode,
            lazy_launch=lazy_launch,
            agent_id=self.agent_id,
        )
