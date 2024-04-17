# -*- coding: utf-8 -*-
"""
AgentScope workstation DAG running engine.

This module defines various workflow nodes that can be used to construct
a computational DAG. Each node represents a step in the DAG and
can perform certain actions when called.
"""
import copy
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List, Any, Optional

import networkx as nx
from loguru import logger

import agentscope
from agentscope.agents import (
    AgentBase,
    DialogAgent,
    UserAgent,
    TextToImageAgent,
    DictDialogAgent,
)
from agentscope.pipelines.functional import placeholder
from agentscope.pipelines import (
    PipelineBase,
    IfElsePipeline,
    SwitchPipeline,
    ForLoopPipeline,
    WhileLoopPipeline,
    SequentialPipeline,
)
from agentscope.msghub import msghub
from agentscope.message import Msg
from agentscope.models import read_model_configs


class WorkflowNodeType(IntEnum):
    """Enum for workflow node."""

    MODEL = 0
    AGENT = 1
    PIPELINE = 2
    SERVICE = 3
    MESSAGE = 4
    COPY = 5


def kwarg_converter(kwargs: dict) -> str:
    """Convert a kwarg dict to a string."""
    kwarg_parts = [f"{key}={repr(value)}" for key, value in kwargs.items()]
    return ", ".join(kwarg_parts)


def deps_converter(dep_vars: list) -> str:
    """Convert a dep_vars list to a string."""
    return f"[{', '.join(dep_vars)}]"


class WorkflowNode(ABC):
    """
    Abstract base class representing a generic node in a workflow.

    WorkflowNode is designed to be subclassed with specific logic implemented
    in the subclass methods. It provides an interface for initialization and
    execution of operations when the node is called.
    """

    node_type = None

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__()
        self.initialize(*args, **kwargs)

    def initialize(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        """
        Initialize nodes. Implement specific initialization logic in
        subclasses.
        """

    @abstractmethod
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        """
        Performs the operations of the node. Implement specific logic in
        subclasses.
        """

    # @abstractmethod
    # def compile(self, var: str):
    #     """
    #     Compile workflow node to python executable code dict
    #     """
    #     return {
    #         "imports": None,
    #         "inits": None,
    #         "execs": None,
    #     }


class ModelNode(WorkflowNode):
    """
    A node that represents a model in a workflow.

    The ModelNode can be used to load and execute a model as part of the
    workflow pipeline. It initializes model configurations and performs
    model-related operations when called.
    """

    node_type = WorkflowNodeType.MODEL

    def initialize(self, **kwargs):  # type: ignore[no-untyped-def]
        self.kwargs = kwargs
        read_model_configs([kwargs])

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    def compile(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        """
        Compile ModelNode to python executable code dict
        """
        return {
            "imports": "from agentscope.models import read_model_configs",
            "inits": f"read_model_configs([{self.kwargs}])",
            "execs": "",
        }


class MsgNode(WorkflowNode):
    """
    A node that manages messaging within a workflow.

    MsgNode is responsible for handling messages, creating message objects,
    and performing message-related operations when the node is invoked.
    """

    node_type = WorkflowNodeType.MESSAGE

    def initialize(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.kwargs = kwargs
        self.msg = Msg(**kwargs)

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self.msg

    def compile(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        """
        Compile ModelNode to python executable code dict
        """
        return {
            "imports": "from agentscope.message import Msg",
            "inits": f"x = Msg({kwarg_converter(self.kwargs)})",
            "execs": "",
        }


class PlaceHolderNode(WorkflowNode):
    """
    A placeholder node within a workflow.

    This node acts as a placeholder and can be used to pass through information
    or data without performing any significant operation.
    """

    node_type = WorkflowNodeType.PIPELINE

    def __call__(self, x: dict = None) -> dict:
        return placeholder(x)


class MsgHubNode(WorkflowNode):
    """
    A node that serves as a messaging hub within a workflow.

    MsgHubNode is responsible for broadcasting announcements to participants
    and managing the flow of messages within a workflow's pipeline.
    """

    node_type = WorkflowNodeType.PIPELINE

    def initialize(  # type: ignore[no-untyped-def]
        self,
        deps: List,
        **kwargs,
    ) -> None:
        self.dep_vars = [x[1] for x in deps]
        deps = [x[0] for x in deps]
        self.announcement = Msg(
            name=kwargs["announcement"].get("name", "Host"),
            content=kwargs["announcement"].get("content", "Welcome!"),
            role="system",
        )
        assert (
            isinstance(deps, list)
            and len(deps) == 1
            and hasattr(
                deps[0],
                "pipeline",
            )
        ), (
            "MsgHub members must be a list of length 1, with the first "
            "element being an instance of PipelineBaseNode"
        )

        self.pipeline = deps[0].pipeline
        self.participants = get_all_agents(self.pipeline)

    def __call__(self, x: dict = None) -> dict:
        with msghub(self.participants, announcement=self.announcement):
            x = self.pipeline(x)
        return x


class SequentialPipelineNode(WorkflowNode):
    """
    A node representing a sequential pipeline within a workflow.

    SequentialPipelineNode executes a series of operators or nodes in a
    sequence, where the output of one node is the input to the next.
    """

    node_type = WorkflowNodeType.PIPELINE

    def initialize(  # type: ignore[no-untyped-def]
        self,
        deps: List,
        **kwargs,
    ) -> None:
        self.dep_vars = [x[1] for x in deps]
        self.deps = deps
        deps = [x[0] for x in deps]
        self.pipeline = SequentialPipeline(operators=deps)

    def __call__(self, x: dict = None) -> dict:
        return self.pipeline(x)

    def compile(self, var: str) -> dict:
        """
        Compile SequentialPipelineNode to python executable code dict
        """
        return {
            "imports": "from agentscope.pipelines import SequentialPipeline",
            "inits": f"{var} = SequentialPipeline("
            f"{deps_converter(self.dep_vars)})",
            "execs": f"x = {var}(x)",
        }


class ForLoopPipelineNode(WorkflowNode):
    """
    A node representing a for-loop structure in a workflow.

    ForLoopPipelineNode allows the execution of a pipeline node multiple times,
    iterating over a given set of inputs or a specified range.
    """

    node_type = WorkflowNodeType.PIPELINE

    def initialize(  # type: ignore[no-untyped-def]
        self,
        deps: List,
        **kwargs,
    ) -> None:
        self.dep_vars = [x[1] for x in deps]
        deps = [x[0] for x in deps]
        assert (
            len(deps) == 1
        ), "ForLoopPipelineNode can only contain one Pipeline Node."
        self.pipeline = ForLoopPipeline(loop_body_operators=deps[0], **kwargs)
        self.kwargs = kwargs

    def __call__(self, x: dict = None) -> dict:
        return self.pipeline(x)

    def compile(self, var: str) -> dict:
        """
        Compile SequentialPipelineNode to python executable code dict
        """
        return {
            "imports": "from agentscope.pipelines import ForLoopPipeline",
            "inits": f"{var} = ForLoopPipeline("
            f"loop_body_operators="
            f"{deps_converter(self.dep_vars)},"
            f" {kwarg_converter(self.kwargs)})",
            "execs": f"x = {var}(x)",
        }


class WhileLoopPipelineNode(WorkflowNode):
    """
    A node representing a while-loop structure in a workflow.

    WhileLoopPipelineNode enables conditional repeated execution of a pipeline
    node based on a specified condition.
    """

    node_type = WorkflowNodeType.PIPELINE

    def initialize(  # type: ignore[no-untyped-def]
        self,
        deps: List,
        **kwargs,
    ) -> None:
        self.dep_vars = [x[1] for x in deps]
        deps = [x[0] for x in deps]
        assert (
            len(deps) == 1
        ), "WhileLoopPipelineNode can only contain one Pipeline Node."
        self.pipeline = WhileLoopPipeline(
            loop_body_operators=deps[0],
            **kwargs,
        )

    def __call__(self, x: dict = None) -> dict:
        return self.pipeline(x)


class IfElsePipelineNode(WorkflowNode):
    """
    A node representing an if-else conditional structure in a workflow.

    IfElsePipelineNode directs the flow of execution to different pipeline
    nodes based on a specified condition.
    """

    node_type = WorkflowNodeType.PIPELINE

    def initialize(  # type: ignore[no-untyped-def]
        self,
        deps: List,
        **kwargs,
    ) -> None:
        self.dep_vars = [x[1] for x in deps]
        deps = [x[0] for x in deps]
        assert (
            0 < len(deps) <= 2
        ), "IfElsePipelineNode must contain one or two Pipeline Node."
        if len(deps) == 1:
            self.pipeline = IfElsePipeline(if_body_operators=deps[0], **kwargs)
        elif len(deps) == 2:
            self.pipeline = IfElsePipeline(
                if_body_operators=deps[0],
                else_body_operators=deps[1],
                **kwargs,
            )

    def __call__(self, x: dict = None) -> dict:
        return self.pipeline(x)


class SwitchPipelineNode(WorkflowNode):
    """
    A node representing a switch-case structure within a workflow.

    SwitchPipelineNode routes the execution to different pipeline nodes
    based on the evaluation of a specified key or condition.
    """

    node_type = WorkflowNodeType.PIPELINE

    def initialize(  # type: ignore[no-untyped-def]
        self,
        deps: List,
        **kwargs,
    ) -> None:
        self.dep_vars = [x[1] for x in deps]
        deps = [x[0] for x in deps]
        assert 0 < len(deps), (
            "SwitchPipelineNode must contain at least " "one Pipeline Node."
        )
        case_operators = {}

        if len(deps) == len(kwargs["cases"]):
            # No default_operators provided
            default_operators = placeholder
        elif len(deps) == len(kwargs["cases"]) + 1:
            # No default_operators provided
            default_operators = deps.pop(-1)
        else:
            raise ValueError(
                f"SwitchPipelineNode node {deps} not matches "
                f"cases {kwargs['cases']}.",
            )

        for key, value in zip(kwargs["cases"], deps):
            case_operators[key] = value
        kwargs.pop("cases")
        self.pipeline = SwitchPipeline(
            case_operators=case_operators,
            default_operators=default_operators,  # type: ignore[arg-type]
            **kwargs,
        )

    def __call__(self, x: dict = None) -> dict:
        return self.pipeline(x)


class CopyNode(WorkflowNode):
    """
    A node that duplicates the output of another node in the workflow.

    CopyNode is used to replicate the results of a parent node and can be
    useful in workflows where the same output is needed for multiple
    subsequent operations.
    """

    node_type = WorkflowNodeType.COPY

    def initialize(  # type: ignore[no-untyped-def]
        self,
        deps: List,
        **kwargs,
    ) -> None:
        self.dep_vars = [x[1] for x in deps]
        deps = [x[0] for x in deps]
        assert len(deps) == 1, "Copy Node can only have one parent!"
        self.pipeline = deps[0]

    def __call__(  # type: ignore[no-untyped-def]
        self,
        *args,
        **kwargs,
    ) -> Any:
        return self.pipeline(*args, **kwargs)


NODE_NAME_MAPPING = {
    "dashscope_chat": (ModelNode, WorkflowNodeType.MODEL),
    "openai_chat": (ModelNode, WorkflowNodeType.MODEL),
    "post_api_chat": (ModelNode, WorkflowNodeType.MODEL),
    "Message": (MsgNode, WorkflowNodeType.MESSAGE),
    "DialogAgent": (DialogAgent, WorkflowNodeType.AGENT),
    "UserAgent": (UserAgent, WorkflowNodeType.AGENT),
    "TextToImageAgent": (TextToImageAgent, WorkflowNodeType.AGENT),
    "DictDialogAgent": (DictDialogAgent, WorkflowNodeType.AGENT),
    "Placeholder": (PlaceHolderNode, WorkflowNodeType.PIPELINE),
    "MsgHub": (MsgHubNode, WorkflowNodeType.PIPELINE),
    "SequentialPipeline": (SequentialPipelineNode, WorkflowNodeType.PIPELINE),
    "ForLoopPipeline": (ForLoopPipelineNode, WorkflowNodeType.PIPELINE),
    "WhileLoopPipeline": (WhileLoopPipelineNode, WorkflowNodeType.PIPELINE),
    "IfElsePipeline": (IfElsePipelineNode, WorkflowNodeType.PIPELINE),
    "SwitchPipeline": (SwitchPipelineNode, WorkflowNodeType.PIPELINE),
    "CopyNode": (CopyNode, WorkflowNodeType.COPY),
}


class ASDiGraph(nx.DiGraph):
    """
    A class that represents a directed graph, extending the functionality of
    networkx's DiGraph to suit specific workflow requirements in AgentScope.

    This graph supports operations such as adding nodes with associated
    computations and executing these computations in a topological order.

    Attributes:
        nodes_not_in_graph (set): A set of nodes that are not included in
        the computation graph.
    """

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        """
        Initialize the ASDiGraph instance.
        """
        super().__init__(*args, **kwargs)
        self.nodes_not_in_graph = set()

        # Prepare the header of the file with necessary imports and any
        # global definitions
        self.imports = [
            "import agentscope",
        ]

        self.inits = [
            """agentscope.init(logger_level="DEBUG")""",
        ]

        self.execs = ["\n"]

    def run(self) -> None:
        """
        Execute the computations associated with each node in the graph.

        The method initializes AgentScope, performs a topological sort of
        the nodes, and then runs each node's computation sequentially using
        the outputs from its predecessors as inputs.
        """
        agentscope.init(logger_level="DEBUG")
        sorted_nodes = list(nx.topological_sort(self))
        sorted_nodes = [
            node_id
            for node_id in sorted_nodes
            if node_id not in self.nodes_not_in_graph
        ]
        logger.info(f"sorted_nodes: {sorted_nodes}")
        logger.info(f"nodes_not_in_graph: {self.nodes_not_in_graph}")

        # Cache output
        values = {}

        # Run with predecessors outputs
        for node_id in sorted_nodes:
            inputs = [
                values[predecessor]
                for predecessor in self.predecessors(node_id)
            ]
            if not inputs:
                values[node_id] = self.exec_node(node_id)
            elif len(inputs):
                # Note: only support exec with the first predecessor now
                values[node_id] = self.exec_node(node_id, inputs[0])
            else:
                raise ValueError("Too many predecessors!")

    def compile(self, compiled_filename: str) -> None:
        """Compile DAG to a runnable python code"""
        sorted_nodes = list(nx.topological_sort(self))
        sorted_nodes = [
            node_id
            for node_id in sorted_nodes
            if node_id not in self.nodes_not_in_graph
        ]

        for node_id in sorted_nodes:
            node = self.nodes[node_id]
            self.execs.append(node["compile_dict"]["execs"])

        header = "\n".join(self.imports)

        body = "\n    ".join(self.inits + self.execs)

        main_body = f"def main():\n    {body}"

        # Combine header and body to form the full script
        script = (
            f"{header}\n\n\n{main_body}\n\nif __name__ == "
            f"'__main__':\n    main()\n"
        )

        # Write the script to file
        with open(compiled_filename, "w", encoding="utf-8") as file:
            file.write(script)

    # pylint: disable=R0912
    def add_as_node(
        self,
        node_id: str,
        node_info: dict,
        config: dict,
    ) -> Any:
        """
        Add a node to the graph based on provided node information and
        configuration.

        Args:
            node_id (str): The identifier for the node being added.
            node_info (dict): A dictionary containing information about the
                node.
            config (dict): Configuration information for the node dependencies.

        Returns:
            The computation object associated with the added node.
        """
        node_cls, node_type = NODE_NAME_MAPPING[node_info.get("name", "")]
        # TODO: support all type of node
        if node_type not in [
            WorkflowNodeType.MODEL,
            WorkflowNodeType.AGENT,
            WorkflowNodeType.MESSAGE,
            WorkflowNodeType.PIPELINE,
            WorkflowNodeType.COPY,
        ]:
            raise NotImplementedError(node_cls)

        if self.has_node(node_id):
            return self.nodes[node_id]["opt"]

        # Init dep nodes
        deps = [str(n) for n in node_info.get("data", {}).get("elements", [])]

        # Exclude for dag when in a Group
        if node_type != WorkflowNodeType.COPY:
            self.nodes_not_in_graph = self.nodes_not_in_graph.union(set(deps))

        dep_opts = []
        for dep_node_id in deps:
            if not self.has_node(dep_node_id):
                dep_node_info = config[dep_node_id]
                self.add_as_node(dep_node_id, dep_node_info, config)
            if isinstance(self.nodes[dep_node_id]["opt"], WorkflowNode):
                node_var = self.nodes[dep_node_id][
                    "opt"
                ].node_type.name.lower()
            elif isinstance(self.nodes[dep_node_id]["opt"], AgentBase):
                node_var = "agent"
            else:
                raise TypeError
            # node_var is for compile
            dep_opts.append(
                (self.nodes[dep_node_id]["opt"], f"{node_var}{dep_node_id}"),
            )

        if len(deps) == 0:
            value = node_cls(**node_info["data"].get("args", {}))
        else:
            value = node_cls(dep_opts, **node_info["data"].get("args", {}))

        # Add build compiled python code
        if isinstance(value, WorkflowNode):
            var_name = value.node_type.name.lower()
            compile_dict = value.compile(var=f"{var_name}{node_id}")
        elif isinstance(value, AgentBase):
            compile_dict = {
                "imports": f"from agentscope.agents import"
                f" {node_info['name']}",
                "inits": f"agent{node_id} = {node_info['name']}"
                f"({kwarg_converter(node_info['data']['args'])})",
                "execs": f"x = agent{node_id}(x)",
            }
        else:
            raise TypeError

        self.add_node(
            node_id,
            opt=value,
            compile_dict=compile_dict,
            **node_info,
        )
        # Insert to imports and inits
        self.imports.append(compile_dict["imports"])

        if (
            hasattr(value, "node_type")
            and value.node_type == WorkflowNodeType.MODEL
        ):
            self.inits.insert(1, compile_dict["inits"])
        else:
            self.inits.append(compile_dict["inits"])
        return value

    def exec_node(self, node_id: str, x_in: Any = None) -> Any:
        """
        Execute the computation associated with a given node in the graph.

        Args:
            node_id (str): The identifier of the node whose computation is
                to be executed.
            x_in: The input to the node's computation. Defaults to None.

        Returns:
            The output of the node's computation.
        """
        logger.debug(
            f"\nnode_id: {node_id}\nin_values:{x_in}",
        )
        opt = self.nodes[node_id]["opt"]
        out_values = opt(x_in)
        logger.debug(
            f"\nnode_id: {node_id}\nout_values:{out_values}",
        )
        return out_values


def get_all_agents(
    pipeline: PipelineBase,
    seen_agents: Optional[set] = None,
) -> List:
    """
    Retrieve all unique agent objects from a pipeline.

    Recursively traverses the pipeline to collect all distinct agent-based
    participants. Prevents duplication by tracking already seen agents.

    Args:
        pipeline (PipelineBase): The pipeline from which to extract agents.
        seen_agents (set, optional): A set of agents that have already been
            seen to avoid duplication. Defaults to None.

    Returns:
        list: A list of unique agent objects found in the pipeline.
    """
    if seen_agents is None:
        seen_agents = set()

    all_agents = []

    for participant in pipeline.participants:
        if isinstance(participant, AgentBase):
            if participant not in seen_agents:
                all_agents.append(participant)
                seen_agents.add(participant)
        elif isinstance(participant, PipelineBase):
            nested_agents = get_all_agents(participant, seen_agents)
            all_agents.extend(nested_agents)

    return all_agents


def sanitize_node_data(raw_info: dict) -> dict:
    """
    Clean and validate node data, evaluating callable expressions where
    necessary.

    Processes the raw node information, removes empty arguments, and evaluates
    any callable expressions provided as string literals.

    Args:
        raw_info (dict): The raw node information dictionary that may contain
            callable expressions as strings.

    Returns:
        dict: The sanitized node information with callable expressions
            evaluated.
    """

    def is_callable_expression(s: str) -> bool:
        try:
            # Do not detect exp like this
            if s in ["input", "print"]:
                return False
            result = eval(s)
            return callable(result)
        except Exception:
            return False

    copied_info = copy.deepcopy(raw_info)
    for key, value in copied_info["data"].get("args", {}).items():
        if not value:
            raw_info["data"]["args"].pop(key)
        elif is_callable_expression(value):
            raw_info["data"]["args"][key] = eval(value)
    return raw_info


def build_dag(config: dict) -> ASDiGraph:
    """
    Construct a Directed Acyclic Graph (DAG) from the provided configuration.

    Initializes the graph nodes based on the configuration, adds model nodes
    first, then non-model nodes, and finally adds edges between the nodes.

    Args:
        config (dict): The configuration to build the graph from, containing
            node info such as name, type, arguments, and connections.

    Returns:
        ASDiGraph: The constructed directed acyclic graph.

    Raises:
        ValueError: If the resulting graph is not acyclic.
    """
    dag = ASDiGraph()

    for node_id, node_info in config.items():
        config[node_id] = sanitize_node_data(node_info)

    # Add and init model nodes first
    for node_id, node_info in config.items():
        if NODE_NAME_MAPPING[node_info["name"]][1] == WorkflowNodeType.MODEL:
            dag.add_as_node(node_id, node_info, config)

    # Add and init non-model nodes
    for node_id, node_info in config.items():
        if NODE_NAME_MAPPING[node_info["name"]][1] != WorkflowNodeType.MODEL:
            dag.add_as_node(node_id, node_info, config)

    # Add edges
    for node_id, node_info in config.items():
        outputs = node_info.get("outputs", {})
        for output_key, output_val in outputs.items():
            connections = output_val.get("connections", [])
            for conn in connections:
                target_node_id = conn.get("node")
                # Here it is assumed that the output of the connection is
                # only connected to one of the inputs. If there are more
                # complex connections, modify the logic accordingly
                dag.add_edge(node_id, target_node_id, output_key=output_key)

    # Check if the graph is a DAG
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("The provided configuration does not form a DAG.")

    return dag
