# -*- coding: utf-8 -*-
""" Client of rpc agent server """

import threading
import json
import os
from typing import Optional, Sequence, Union, Tuple
from loguru import logger

try:
    import dill
    import grpc
    from grpc import RpcError
    from google.protobuf.empty_pb2 import Empty
    from agentscope.rpc.rpc_agent_pb2_grpc import RpcAgentStub
    import agentscope.rpc.rpc_agent_pb2 as agent_pb2
except ImportError as import_error:
    from agentscope.utils.tools import ImportErrorReporter

    dill = ImportErrorReporter(import_error, "distribute")
    grpc = ImportErrorReporter(import_error, "distribute")
    agent_pb2 = ImportErrorReporter(import_error, "distribute")
    RpcAgentStub = ImportErrorReporter(import_error, "distribute")
    RpcError = ImportError

from agentscope.file_manager import file_manager
from agentscope.utils.tools import generate_id_from_seed
from agentscope.exception import AgentServerNotAliveError
from agentscope.constants import _DEFAULT_RPC_OPTIONS


class RpcAgentClient:
    """A client of Rpc agent server"""

    def __init__(
        self,
        host: str,
        port: int,
        agent_id: str = None,
    ) -> None:
        """Init a rpc agent client

        Args:
            host (`str`): The hostname of the rpc agent server which the
            client is connected.
            port (`int`): The port of the rpc agent server which the client
            is connected.
            agent_id (`str`): The agent id of the agent being called.
            Defaults to None.
        """
        self.host = host
        self.port = port
        self.agent_id = agent_id

    def call_agent_func(
        self,
        func_name: str,
        value: Optional[str] = None,
        timeout: int = 300,
    ) -> str:
        """Call the specific function of an agent running on the server.

        Args:
            func_name (`str`): The name of the function being called.
            valuer (`str`, optional): The serialized function input value.
            Defaults to None.
            timeout (`int`, optional): The timeout for the RPC call in seconds.
            Defaults to 300.

        Returns:
            str: serialized return data.
        """
        try:
            with grpc.insecure_channel(f"{self.host}:{self.port}") as channel:
                stub = RpcAgentStub(channel)
                result_msg = stub.call_agent_func(
                    agent_pb2.RpcMsg(
                        value=value,
                        target_func=func_name,
                        agent_id=self.agent_id,
                    ),
                    timeout=timeout,
                )
                return result_msg.message
        except Exception as e:
            # check the server and raise a more reasonable error
            if not self.is_alive():
                raise AgentServerNotAliveError(
                    host=self.host,
                    port=self.port,
                    message=str(e),
                ) from e
            raise e

    def is_alive(self) -> bool:
        """Check if the agent server is alive.

        Returns:
            bool: Indecate whether the server is alive.
        """

        try:
            with grpc.insecure_channel(f"{self.host}:{self.port}") as channel:
                stub = RpcAgentStub(channel)
                status = stub.is_alive(Empty(), timeout=5)
                if not status.ok:
                    raise AgentServerNotAliveError(
                        host=self.host,
                        port=self.port,
                    )
                return status.ok
        except Exception:
            logger.info(
                f"Agent server [{self.host}:{self.port}] not alive.",
            )
            return False

    def stop(self) -> None:
        """Stop the agent server."""
        try:
            with grpc.insecure_channel(f"{self.host}:{self.port}") as channel:
                stub = RpcAgentStub(channel)
                logger.info(
                    f"Stopping agent server at [{self.host}:{self.port}].",
                )
                resp = stub.stop(Empty(), timeout=5)
                if resp.ok:
                    logger.info(
                        f"Agent server at [{self.host}:{self.port}] stopped.",
                    )
                else:
                    logger.error(
                        f"Fail to stop the agent server: {resp.message}",
                    )
        except Exception as e:
            logger.error(
                f"Fail to stop the agent server: {e}",
            )

    def create_agent(
        self,
        agent_configs: dict,
        agent_id: str = None,
    ) -> bool:
        """Create a new agent for this client.

        Args:
            agent_configs (`dict`): Init configs of the agent, generated by
            `_AgentMeta`.
            agent_id (`str`): agent_id of the created agent.

        Returns:
            bool: Indecate whether the creation is successful
        """
        try:
            with grpc.insecure_channel(f"{self.host}:{self.port}") as channel:
                stub = RpcAgentStub(channel)
                status = stub.create_agent(
                    agent_pb2.CreateAgentRequest(
                        agent_id=(
                            self.agent_id if agent_id is None else agent_id
                        ),
                        agent_init_args=dill.dumps(agent_configs),
                    ),
                )
                if not status.ok:
                    logger.error(
                        f"Error when creating agent: {status.message}",
                    )
                return status.ok
        except Exception as e:
            # check the server and raise a more reasonable error
            if not self.is_alive():
                raise AgentServerNotAliveError(
                    host=self.host,
                    port=self.port,
                    message=str(e),
                ) from e
            raise e

    def delete_agent(
        self,
        agent_id: str = None,
    ) -> bool:
        """
        Delete agents with the specific agent_id.

        Args:
            agent_id (`str`): id of the agent to be deleted.

        Returns:
            bool: Indecate whether the deletion is successful
        """
        with grpc.insecure_channel(f"{self.host}:{self.port}") as channel:
            stub = RpcAgentStub(channel)
            status = stub.delete_agent(
                agent_pb2.StringMsg(value=agent_id),
            )
            if not status.ok:
                logger.error(f"Error when deleting agent: {status.message}")
            return status.ok

    def delete_all_agent(self) -> bool:
        """Delete all agents on the server."""
        with grpc.insecure_channel(f"{self.host}:{self.port}") as channel:
            stub = RpcAgentStub(channel)
            status = stub.delete_all_agents(Empty())
            if not status.ok:
                logger.error(f"Error when delete all agents: {status.message}")
            return status.ok

    def clone_agent(self, agent_id: str) -> Optional[str]:
        """Clone a new agent instance from the origin instance.

        Args:
            agent_id (`str`): The agent_id of the agent to be cloned.

        Returns:
            str: The `agent_id` of the generated agent.
        """
        with grpc.insecure_channel(f"{self.host}:{self.port}") as channel:
            stub = RpcAgentStub(channel)
            resp = stub.clone_agent(
                agent_pb2.StringMsg(value=agent_id),
            )
            if not resp.ok:
                logger.error(
                    f"Error when clone agent [{agent_id}]: {resp.message}",
                )
                return None
            else:
                return resp.message

    def update_placeholder(self, task_id: int) -> Tuple[bool, str]:
        """Update the placeholder value.

        Args:
            task_id (`int`): `task_id` of the PlaceholderMessage.

        Returns:
            bool: Whether the update is successful.
            str: Serialized message value.
        """
        with grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=_DEFAULT_RPC_OPTIONS,
        ) as channel:
            stub = RpcAgentStub(channel)
            result_msg = stub.update_placeholder(
                agent_pb2.UpdatePlaceholderRequest(task_id=task_id),
            )
            return result_msg.ok, result_msg.message

    def get_agent_list(self) -> Sequence[dict]:
        """
        Get the summary of all agents on the server as a list.

        Returns:
            Sequence[str]: list of agent summary information.
        """
        with grpc.insecure_channel(f"{self.host}:{self.port}") as channel:
            stub = RpcAgentStub(channel)
            resp = stub.get_agent_list(Empty())
            return [
                json.loads(agent_str) for agent_str in json.loads(resp.message)
            ]

    def get_server_info(self) -> dict:
        """Get the agent server resource usage information."""
        try:
            with grpc.insecure_channel(f"{self.host}:{self.port}") as channel:
                stub = RpcAgentStub(channel)
                resp = stub.get_server_info(Empty())
                if not resp.ok:
                    logger.error(f"Error in get_server_info: {resp.message}")
                    return {}
                return json.loads(resp.message)
        except Exception:
            logger.info(f"Agent server [{self.host}:{self.port}] not alive.")
            return {}

    def set_model_configs(
        self,
        model_configs: Union[dict, list[dict]],
    ) -> bool:
        """Set the model configs of the server."""
        with grpc.insecure_channel(f"{self.host}:{self.port}") as channel:
            stub = RpcAgentStub(channel)
            resp = stub.set_model_configs(
                agent_pb2.StringMsg(value=json.dumps(model_configs)),
            )
            if not resp.ok:
                logger.error(f"Error in set_model_configs: {resp.message}")
                return False
            return True

    def get_agent_memory(self, agent_id: str) -> Union[list, dict]:
        """Get the memory usage of the specific agent."""
        with grpc.insecure_channel(f"{self.host}:{self.port}") as channel:
            stub = RpcAgentStub(channel)
            resp = stub.get_agent_memory(
                agent_pb2.StringMsg(value=agent_id),
            )
            if not resp.ok:
                logger.error(f"Error in get_agent_memory: {resp.message}")
            return json.loads(resp.message)

    def download_file(self, path: str) -> str:
        """Download a file from a remote server to the local machine.

        Args:
        path (`str`): The path of the file to be downloaded. Note that
            it is the path on the remote server.

        Returns:
            `str`: The path of the downloaded file. Note that it is the path
            on the local machine.
        """
        local_file_name = (
            f"{generate_id_from_seed(path, 5)}_{os.path.basename(path)}"
        )
        local_path = os.path.join(file_manager.dir_file, local_file_name)
        with grpc.insecure_channel(f"{self.host}:{self.port}") as channel:
            stub = RpcAgentStub(channel)
            with open(local_path, "wb") as f:
                for resp in stub.download_file(
                    agent_pb2.StringMsg(value=path),
                ):
                    f.write(resp.data)
            return local_path


class ResponseStub:
    """A stub used to save the response of an rpc call in a sub-thread."""

    def __init__(self) -> None:
        self.response = None
        self.condition = threading.Condition()

    def set_response(self, response: str) -> None:
        """Set the message."""
        with self.condition:
            self.response = response
            self.condition.notify_all()

    def get_response(self) -> str:
        """Get the message."""
        with self.condition:
            while self.response is None:
                self.condition.wait()
            return self.response


def call_in_thread(
    client: RpcAgentClient,
    value: str,
    func_name: str,
) -> ResponseStub:
    """Call rpc function in a sub-thread.

    Args:
        client (`RpcAgentClient`): The rpc client.
        x (`str`): The value of the reqeust.
        func_name (`str`): The name of the function being called.

    Returns:
        `ResponseStub`: A stub to get the response.
    """
    stub = ResponseStub()

    def wrapper() -> None:
        try:
            resp = client.call_agent_func(
                func_name=func_name,
                value=value,
            )
            stub.set_response(resp)  # type: ignore[arg-type]
        except RpcError as e:
            logger.error(f"Fail to call {func_name} in thread: {e}")
            stub.set_response(str(e))

    thread = threading.Thread(target=wrapper)
    thread.start()
    return stub
