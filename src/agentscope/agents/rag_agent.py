# -*- coding: utf-8 -*-
"""
This example shows how to build an agent with RAG
with LlamaIndex.

Notice, this is a Beta version of RAG agent.
"""

from typing import Any, Optional, Union, Sequence
from loguru import logger

from agentscope.agents.agent import AgentBase
from agentscope.message import Msg
from agentscope.rag import Knowledge

from agentscope.utils.common import _convert_to_str

from agentscope.rag.llama_index_knowledge import LlamaIndexKnowledge
from agentscope.rag.langchain_knowledge import LangChainKnowledge

CHECKING_PROMPT = """
                Is the retrieved content relevant to the query?
                Retrieved content: {}
                Query: {}
                Only answer YES or NO.
                """


class RAG_Agent(AgentBase):
    """
    A RAG agent build on LlamaIndex or Langchain.
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        knowledge_list: list[Knowledge] = None,
        knowledge_id_list: list[str] = None,
        similarity_top_k: int = None,
        search_type: str = "similarity",
        search_kwargs: dict = None,
        log_retrieval: bool = True,
        recent_n_mem_for_retrieve: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the RAG LlamaIndexAgent
        Args:
            name (str):
                the name for the agent
            sys_prompt (str):
                system prompt for the RAG agent
            model_config_name (str):
                language model for the agent
            knowledge_list (list[Knowledge]):
                a list of knowledge.
                User can choose to pass a list knowledge object
                directly when initializing the RAG agent. Another
                choice can be passing a list of knowledge ids and
                obtain the knowledge with the `equip` function of a
                knowledge bank.
            knowledge_id_list (list[Knowledge]):
                a list of id of the knowledge.
                This is designed for easy setting up multiple RAG
                agents with a config file. To obtain the knowledge
                objects, users can pass this agent to the `equip`
                function in a knowledge bank to add corresponding
                knowledge to agent's self.knowledge_list.
            similarity_top_k (int):
                the number of most similar data blocks retrieved
                from each of the knowledge
            search_type (str):
                the type of search to be performed on the
                Langchain knowledge
            search_kwargs (dict):
                additional keyword arguments for the
                search operation on the Langchain knowledge
            log_retrieval (bool):
                whether to print the retrieved content
            recent_n_mem_for_retrieve (int):
                the number of pieces of memory used as part of
                retrival query
        """
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )
        self.knowledge_list = knowledge_list or []
        self.knowledge_id_list = knowledge_id_list or []
        self.similarity_top_k = similarity_top_k
        self.search_type = search_type
        self.search_kwargs = search_kwargs or {}
        self.log_retrieval = log_retrieval
        self.recent_n_mem_for_retrieve = recent_n_mem_for_retrieve
        self.description = kwargs.get("description", "")

    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        """
        Reply function of the RAG agent.
        Processes the input data,
        1) use the input data to retrieve with RAG function;
        2) generates a prompt using the current memory and system
        prompt;
        3) invokes the language model to produce a response. The
        response is then formatted and added to the dialogue memory.

        Args:
            x (`Optional[Union[Msg, Sequence[Msg]]]`, defaults to `None`):
                The input message(s) to the agent, which also can be omitted if
                the agent doesn't need any input.

        Returns:
            `Msg`: The output message generated by the agent.
        """
        retrieved_docs_to_strings = ""
        # record the input if needed
        if self.memory:
            self.memory.add(x)
            # in case no input is provided (e.g., in msghub),
            # use the memory as query
            history = self.memory.get_memory(
                recent_n=self.recent_n_mem_for_retrieve,
            )
            query = (
                "/n".join(
                    [msg.content for msg in history],
                )
                if isinstance(history, list)
                else str(history)
            )
        elif x is not None:
            query = x.content
        else:
            query = ""

        if len(query) > 0:
            # when content has information, do retrieval
            for knowledge in self.knowledge_list:
                retrieved_nodes = knowledge.retrieve(
                    str(query),
                    self.similarity_top_k,
                    search_type=self.search_type,
                    search_kwargs=self.search_kwargs,
                )
                if isinstance(knowledge, LlamaIndexKnowledge):
                    retrieved_docs_to_strings += (
                        self._llama_index_parse_retrieved_nodes(
                            retrieved_nodes,
                            query,
                        )
                    )
                elif isinstance(knowledge, LangChainKnowledge):
                    retrieved_docs_to_strings += (
                        self._langchain_parse_retrieved_nodes(
                            retrieved_nodes,
                        )
                    )
                else:
                    raise ValueError("Unknown knowledge type.")

            if self.log_retrieval:
                self.speak("[retrieved]:" + retrieved_docs_to_strings)

        # prepare prompt
        prompt = self.model.format(
            Msg(
                name="system",
                role="system",
                content=self.sys_prompt,
            ),
            # {"role": "system", "content": retrieved_docs_to_string},
            self.memory.get_memory(
                recent_n=self.recent_n_mem_for_retrieve,
            ),
            Msg(
                name="user",
                role="user",
                content="Context: " + retrieved_docs_to_strings,
            ),
        )

        # call llm and generate response
        response = self.model(prompt).text
        msg = Msg(self.name, response, "assistant")

        # Print/speak the message in this agent's voice
        self.speak(msg)

        if self.memory:
            # Record the message in memory
            self.memory.add(msg)

        return msg

    def _llama_index_parse_retrieved_nodes(
        self,
        retrieved_nodes: list,
        query: str,
    ) -> str:
        """
        Parses the retrieved nodes from LlamaIndexand and formats them
        into a string representation.

        Processes the retrieved nodes by concatenating their scores, sources,
        and contents into a single string. If the maximum score is below a
        threshold (0.4), it uses a language model to determine if the retrieved
        content is relevant to the user's query.
        """
        retrieved_docs_to_string = ""
        scores = []
        for node in retrieved_nodes:
            scores.append(node.score)
            retrieved_docs_to_string += (
                "\n>>>> score:"
                + str(node.score)
                + "\n>>>> source:"
                + str(node.node.get_metadata_str())
                + "\n>>>> content:"
                + node.get_content()
            )
        if max(scores) < 0.4:
            # if the max score is lower than 0.4, then we let LLM
            # decide whether the retrieved content is relevant
            # to the user input.
            msg = Msg(
                name="user",
                role="user",
                content=CHECKING_PROMPT.format(
                    retrieved_docs_to_string,
                    query,
                ),
            )
            msg = self.model.format(msg)
            checking = self.model(msg)
            logger.info(checking)
            checking = checking.text.lower()
            if "no" in checking:
                retrieved_docs_to_string = "EMPTY"
        return retrieved_docs_to_string

    def _langchain_parse_retrieved_nodes(self, retrieved_nodes: list) -> str:
        """
        Parses the retrieved nodes from langchain and
        formats them into a string.

        Processes the retrieved documents by concatenating their sources and
        contents into a single string.
        """

        retrieved_docs_to_string = ""
        for document in retrieved_nodes:
            retrieved_docs_to_string += (
                "\n>>>> source:"
                + _convert_to_str(document.metadata)
                + "\n>>>> content:"
                + document.page_content
            )
        return retrieved_docs_to_string


LlamaIndexAgent = RAG_Agent
