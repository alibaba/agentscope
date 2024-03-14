# -*- coding: utf-8 -*-
"""
This example shows how to build a agent with RAG (backup by LlamaIndex)
"""

from typing import Optional
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

from agentscope.prompt import PromptType
from agentscope.agents.agent import AgentBase
from agentscope.prompt import PromptEngine
from agentscope.message import Msg
from agentscope.rag.llama_index_rag import LlamaIndexRAG


Settings.embed_model = DashScopeEmbedding()


class LlamaIndexAgent(AgentBase):
    """
    A LlamaIndex agent build on LlamaIndex.
    """

    def __init__(
        self,
        name: str,
        sys_prompt: Optional[str] = None,
        model_config_name: str = None,
        use_memory: bool = True,
        memory_config: Optional[dict] = None,
        prompt_type: Optional[PromptType] = PromptType.LIST,
    ) -> None:
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            use_memory=use_memory,
            memory_config=memory_config,
        )
        # init prompt engine
        self.engine = PromptEngine(self.model, prompt_type=prompt_type)
        self.rag = LlamaIndexRAG(
            model=self.model,
            loader_type=SimpleDirectoryReader,
            vector_store_type=VectorStoreIndex,
        )
        docs = self.rag.load_data(path="./data")
        self.rag.store_and_index(docs)

    def reply(self, x: dict = None) -> dict:
        """
        Reply function of the LlamaIndex agent.
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
        # record the input if needed
        if x is not None:
            self.memory.add(x)

        content = x.content
        retrieved_docs = self.rag.retrieve(content)

        retrieved_docs_to_string = ""
        for node in retrieved_docs:
            print(node)
            retrieved_docs_to_string += node.get_text()

        print(retrieved_docs_to_string)

        # prepare prompt
        prompt = self.engine.join(
            {
                "role": "system",
                "content": self.sys_prompt.format_map(
                    {"retrieved_context": retrieved_docs_to_string},
                ),
            },
            # {"role": "system", "content": retrieved_docs_to_string},
            self.memory.get_memory(),
        )

        print(prompt)
        # call llm and generate response
        response = self.model(prompt).text
        msg = Msg(self.name, response)

        # Print/speak the message in this agent's voice
        self.speak(msg)

        # Record the message in memory
        self.memory.add(msg)

        return msg
