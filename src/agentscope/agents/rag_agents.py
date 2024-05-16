# -*- coding: utf-8 -*-
"""
This example shows how to build an agent with RAG
with LlamaIndex.

Notice, this is a Beta version of RAG agent.
"""

from typing import Optional, Any
from loguru import logger

from agentscope.agents.agent import AgentBase
from agentscope.message import Msg
from agentscope.rag import Knowledge


CHECKING_PROMPT = """
                Does the retrieved content is relevant to the query?
                Retrieved content: {}
                Query: {}
                Only answer YES or NO.
                """


class LlamaIndexAgent(AgentBase):
    """
    A LlamaIndex agent build on LlamaIndex.
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        knowledge_list: list[Knowledge] = None,
        memory_config: Optional[dict] = None,
        rag_config: Optional[dict] = None,
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
            memory_config (dict):
                memory configuration
            rag_config (dict):
                config for RAG module. It contains at least
                the following parameters:
                "knowledge_id" (str):
                    identifier of the knowledge in KnowledgeBank,
                "similarity_top_k" (int):
                    how many nodes/document to retrieved,
                "log_retrieval" (bool):
                    whether log the retrieved content,
                "recent_n_mem" (int):
                    how many memory used to query (default is 1,
                    using only the current input to reply)
        """
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            memory_config=memory_config,
        )
        self.knowledge_list = knowledge_list or []
        self.retriever_list = []
        self.description = kwargs.get("description", "")
        self.rag_config = rag_config or {}

    def reply(self, x: dict = None) -> dict:
        """
        Reply function of the RAG agent.
        Processes the input data,
        1) use the input data to retrieve with RAG function;
        2) generates a prompt using the current memory and system
        prompt;
        3) invokes the language model to produce a response. The
        response is then formatted and added to the dialogue memory.

        Args:
            x (`dict`, defaults to `None`):
                A dictionary representing the user's input to the agent. This
                input is added to the memory if provided. Defaults to
                None.
        Returns:
            A dictionary representing the message generated by the agent in
            response to the user's input.
        """
        retrieved_docs_to_string = ""
        # record the input if needed
        if self.memory:
            self.memory.add(x)
            # in case no input is provided (e.g., in msghub),
            # use the memory as query
            history = self.memory.get_memory(
                recent_n=self.rag_config.get("recent_n_mem", 1),
            )
            query = (
                "/n".join(
                    [msg["content"] for msg in history],
                )
                if isinstance(history, list)
                else str(history)
            )
        elif x is not None:
            query = x["content"]
        else:
            query = ""

        if len(query) > 0:
            # when content has information, do retrieval
            scores = []
            for retriever in self.retriever_list:
                retrieved_nodes = retriever.retrieve(str(query))
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

            if self.rag_config["log_retrieval"]:
                self.speak("[retrieved]:" + retrieved_docs_to_string)

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
                print(msg)
                checking = self.model([msg])
                logger.info(checking)
                checking = checking.text.lower()
                if "no" in checking:
                    retrieved_docs_to_string = "EMPTY"

        # prepare prompt
        prompt = self.model.format(
            Msg(
                name="system",
                role="system",
                content=self.sys_prompt,
            ),
            # {"role": "system", "content": retrieved_docs_to_string},
            self.memory.get_memory(
                recent_n=self.rag_config.get("recent_n_mem", 1),
            ),
            Msg(
                name="user",
                role="user",
                content="Context: " + retrieved_docs_to_string,
            ),
        )

        # call llm and generate response
        response = self.model(prompt).text
        msg = Msg(self.name, response)

        # Print/speak the message in this agent's voice
        self.speak(msg)

        if self.memory:
            # Record the message in memory
            self.memory.add(msg)

        return msg
