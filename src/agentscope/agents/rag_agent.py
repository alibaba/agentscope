# -*- coding: utf-8 -*-
"""
This example shows how to build an agent with RAG
with LlamaIndex.

Notice, this is a Beta version of RAG agent.
"""
import json
from typing import Any, Optional, Union, Sequence
from loguru import logger

from agentscope.agents.agent import AgentBase
from agentscope.message import Msg
from agentscope.rag import Knowledge

CHECKING_PROMPT = """
                Is the retrieved content relevant to the query?
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
        knowledge_id_list: list[str] = None,
        similarity_top_k: int = None,
        log_retrieval: bool = True,
        recent_n_mem_for_retrieve: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the RAG LlamaIndexAgent
        Args:
            name (str):
                The name for the agent
            sys_prompt (str):
                System prompt for the RAG agent
            model_config_name (str):
                Language model for the agent
            knowledge_list (list[Knowledge]):
                A list of knowledge.
                User can choose to pass a list knowledge object
                directly when initializing the RAG agent. Another
                choice can be passing a list of knowledge ids and
                obtain the knowledge with the `equip` function of a
                knowledge bank.
            knowledge_id_list (list[Knowledge]):
                A list of id of the knowledge.
                This is designed for easy setting up multiple RAG
                agents with a config file. To obtain the knowledge
                objects, users can pass this agent to the `equip`
                function in a knowledge bank to add corresponding
                knowledge to agent's self.knowledge_list.
            similarity_top_k (int):
                The number of most similar data blocks retrieved
                from each of the knowledge
            log_retrieval (bool):
                Whether to print the retrieved content
            recent_n_mem_for_retrieve (int):
                The number of pieces of memory used as part of
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
        retrieved_docs_to_string = ""
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
            scores = []
            for knowledge in self.knowledge_list:
                retrieved_chunks = knowledge.retrieve(
                    str(query),
                    self.similarity_top_k,
                )
                for chunk in retrieved_chunks:
                    scores.append(chunk.score)
                    retrieved_docs_to_string += (
                        json.dumps(
                            chunk.to_dict(),
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n"
                    )

            if self.log_retrieval:
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
                msg = self.model.format(msg)
                checking = self.model(msg)
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
                recent_n=self.recent_n_mem_for_retrieve,
            ),
            Msg(
                name="user",
                role="user",
                content="Context: " + retrieved_docs_to_string,
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
