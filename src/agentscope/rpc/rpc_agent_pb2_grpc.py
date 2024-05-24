# -*- coding: utf-8 -*-
# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
try:
    import grpc
    from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
except ImportError as import_error:
    from agentscope.utils.tools import ImportErrorReporter

    grpc = ImportErrorReporter(import_error, "distribute")
    google_dot_protobuf_dot_empty__pb2 = ImportErrorReporter(
        import_error,
        "distribute",
    )
import agentscope.rpc.rpc_agent_pb2 as rpc__agent__pb2


class RpcAgentStub(object):
    """Servicer for rpc agent server"""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.is_alive = channel.unary_unary(
            "/RpcAgent/is_alive",
            request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            response_deserializer=rpc__agent__pb2.StatusResponse.FromString,
        )
        self.create_agent = channel.unary_unary(
            "/RpcAgent/create_agent",
            request_serializer=rpc__agent__pb2.CreateAgentRequest.SerializeToString,
            response_deserializer=rpc__agent__pb2.StatusResponse.FromString,
        )
        self.delete_agent = channel.unary_unary(
            "/RpcAgent/delete_agent",
            request_serializer=rpc__agent__pb2.AgentIds.SerializeToString,
            response_deserializer=rpc__agent__pb2.StatusResponse.FromString,
        )
        self.clone_agent = channel.unary_unary(
            "/RpcAgent/clone_agent",
            request_serializer=rpc__agent__pb2.AgentIds.SerializeToString,
            response_deserializer=rpc__agent__pb2.AgentIds.FromString,
        )
        self.get_agent_id_list = channel.unary_unary(
            "/RpcAgent/get_agent_id_list",
            request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            response_deserializer=rpc__agent__pb2.AgentIds.FromString,
        )
        self.get_agent_info = channel.unary_unary(
            "/RpcAgent/get_agent_info",
            request_serializer=rpc__agent__pb2.AgentIds.SerializeToString,
            response_deserializer=rpc__agent__pb2.StatusResponse.FromString,
        )
        self.get_server_info = channel.unary_unary(
            "/RpcAgent/get_server_info",
            request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            response_deserializer=rpc__agent__pb2.StatusResponse.FromString,
        )
        self.set_model_configs = channel.unary_unary(
            "/RpcAgent/set_model_configs",
            request_serializer=rpc__agent__pb2.JsonMsg.SerializeToString,
            response_deserializer=rpc__agent__pb2.StatusResponse.FromString,
        )
        self.get_agent_memory = channel.unary_unary(
            "/RpcAgent/get_agent_memory",
            request_serializer=rpc__agent__pb2.AgentIds.SerializeToString,
            response_deserializer=rpc__agent__pb2.JsonMsg.FromString,
        )
        self.call_agent_func = channel.unary_unary(
            "/RpcAgent/call_agent_func",
            request_serializer=rpc__agent__pb2.RpcMsg.SerializeToString,
            response_deserializer=rpc__agent__pb2.RpcMsg.FromString,
        )
        self.update_placeholder = channel.unary_unary(
            "/RpcAgent/update_placeholder",
            request_serializer=rpc__agent__pb2.UpdatePlaceholderRequest.SerializeToString,
            response_deserializer=rpc__agent__pb2.RpcMsg.FromString,
        )
        self.download_file = channel.unary_stream(
            "/RpcAgent/download_file",
            request_serializer=rpc__agent__pb2.FileRequest.SerializeToString,
            response_deserializer=rpc__agent__pb2.FileResponse.FromString,
        )


class RpcAgentServicer(object):
    """Servicer for rpc agent server"""

    def is_alive(self, request, context):
        """check server is alive"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def create_agent(self, request, context):
        """create a new agent on the server"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def delete_agent(self, request, context):
        """delete agents from the server"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def clone_agent(self, request, context):
        """clone an agent with specific agent_id"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def get_agent_id_list(self, request, context):
        """get id of all agents on the server as a list"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def get_agent_info(self, request, context):
        """get the agent information of the specific agent_id"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def get_server_info(self, request, context):
        """get the resource utilization information of the server"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def set_model_configs(self, request, context):
        """update the model configs in the server"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def get_agent_memory(self, request, context):
        """get memory of specific agent"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def call_agent_func(self, request, context):
        """call funcs of agent running on the server"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def update_placeholder(self, request, context):
        """update value of PlaceholderMessage"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def download_file(self, request, context):
        """file transfer"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_RpcAgentServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "is_alive": grpc.unary_unary_rpc_method_handler(
            servicer.is_alive,
            request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            response_serializer=rpc__agent__pb2.StatusResponse.SerializeToString,
        ),
        "create_agent": grpc.unary_unary_rpc_method_handler(
            servicer.create_agent,
            request_deserializer=rpc__agent__pb2.CreateAgentRequest.FromString,
            response_serializer=rpc__agent__pb2.StatusResponse.SerializeToString,
        ),
        "delete_agent": grpc.unary_unary_rpc_method_handler(
            servicer.delete_agent,
            request_deserializer=rpc__agent__pb2.AgentIds.FromString,
            response_serializer=rpc__agent__pb2.StatusResponse.SerializeToString,
        ),
        "clone_agent": grpc.unary_unary_rpc_method_handler(
            servicer.clone_agent,
            request_deserializer=rpc__agent__pb2.AgentIds.FromString,
            response_serializer=rpc__agent__pb2.AgentIds.SerializeToString,
        ),
        "get_agent_id_list": grpc.unary_unary_rpc_method_handler(
            servicer.get_agent_id_list,
            request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            response_serializer=rpc__agent__pb2.AgentIds.SerializeToString,
        ),
        "get_agent_info": grpc.unary_unary_rpc_method_handler(
            servicer.get_agent_info,
            request_deserializer=rpc__agent__pb2.AgentIds.FromString,
            response_serializer=rpc__agent__pb2.StatusResponse.SerializeToString,
        ),
        "get_server_info": grpc.unary_unary_rpc_method_handler(
            servicer.get_server_info,
            request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            response_serializer=rpc__agent__pb2.StatusResponse.SerializeToString,
        ),
        "set_model_configs": grpc.unary_unary_rpc_method_handler(
            servicer.set_model_configs,
            request_deserializer=rpc__agent__pb2.JsonMsg.FromString,
            response_serializer=rpc__agent__pb2.StatusResponse.SerializeToString,
        ),
        "get_agent_memory": grpc.unary_unary_rpc_method_handler(
            servicer.get_agent_memory,
            request_deserializer=rpc__agent__pb2.AgentIds.FromString,
            response_serializer=rpc__agent__pb2.JsonMsg.SerializeToString,
        ),
        "call_agent_func": grpc.unary_unary_rpc_method_handler(
            servicer.call_agent_func,
            request_deserializer=rpc__agent__pb2.RpcMsg.FromString,
            response_serializer=rpc__agent__pb2.RpcMsg.SerializeToString,
        ),
        "update_placeholder": grpc.unary_unary_rpc_method_handler(
            servicer.update_placeholder,
            request_deserializer=rpc__agent__pb2.UpdatePlaceholderRequest.FromString,
            response_serializer=rpc__agent__pb2.RpcMsg.SerializeToString,
        ),
        "download_file": grpc.unary_stream_rpc_method_handler(
            servicer.download_file,
            request_deserializer=rpc__agent__pb2.FileRequest.FromString,
            response_serializer=rpc__agent__pb2.FileResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "RpcAgent",
        rpc_method_handlers,
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class RpcAgent(object):
    """Servicer for rpc agent server"""

    @staticmethod
    def is_alive(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/RpcAgent/is_alive",
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            rpc__agent__pb2.StatusResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def create_agent(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/RpcAgent/create_agent",
            rpc__agent__pb2.CreateAgentRequest.SerializeToString,
            rpc__agent__pb2.StatusResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def delete_agent(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/RpcAgent/delete_agent",
            rpc__agent__pb2.AgentIds.SerializeToString,
            rpc__agent__pb2.StatusResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def clone_agent(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/RpcAgent/clone_agent",
            rpc__agent__pb2.AgentIds.SerializeToString,
            rpc__agent__pb2.AgentIds.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def get_agent_id_list(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/RpcAgent/get_agent_id_list",
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            rpc__agent__pb2.AgentIds.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def get_agent_info(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/RpcAgent/get_agent_info",
            rpc__agent__pb2.AgentIds.SerializeToString,
            rpc__agent__pb2.StatusResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def get_server_info(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/RpcAgent/get_server_info",
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            rpc__agent__pb2.StatusResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def set_model_configs(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/RpcAgent/set_model_configs",
            rpc__agent__pb2.JsonMsg.SerializeToString,
            rpc__agent__pb2.StatusResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def get_agent_memory(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/RpcAgent/get_agent_memory",
            rpc__agent__pb2.AgentIds.SerializeToString,
            rpc__agent__pb2.JsonMsg.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def call_agent_func(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/RpcAgent/call_agent_func",
            rpc__agent__pb2.RpcMsg.SerializeToString,
            rpc__agent__pb2.RpcMsg.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def update_placeholder(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/RpcAgent/update_placeholder",
            rpc__agent__pb2.UpdatePlaceholderRequest.SerializeToString,
            rpc__agent__pb2.RpcMsg.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def download_file(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_stream(
            request,
            target,
            "/RpcAgent/download_file",
            rpc__agent__pb2.FileRequest.SerializeToString,
            rpc__agent__pb2.FileResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
