# -*- coding: utf-8 -*-
"""
This example shows how to build an agent with RAG (backup by LlamaIndex)
"""

from typing import Optional
from llama_index.core import SimpleDirectoryReader
from langchain_community.document_loaders import DirectoryLoader

from agentscope.prompt import PromptType
from agentscope.agents.agent import AgentBase
from agentscope.prompt import PromptEngine
from agentscope.message import Msg
from agentscope.models import load_model_by_config_name
from agentscope.rag.llama_index_rag import LlamaIndexRAG
from agentscope.rag.langchain_rag import LangChainRAG


class RAGAgent(AgentBase):
    """
    Base class for RAG agents, child classes include the
    RAG agents built with LlamaIndex and LangChain in this file
    """

    def __init__(
        self,
        name: str,
        sys_prompt: Optional[str] = None,
        model_config_name: str = None,
        emb_model_config_name: str = None,
        use_memory: bool = True,
        memory_config: Optional[dict] = None,
        prompt_type: Optional[PromptType] = PromptType.LIST,
        config: Optional[dict] = None,
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
        self.emb_model = load_model_by_config_name(emb_model_config_name)

        # init rag as None
        # MUST USE LlamaIndexAgent OR LangChainAgent
        self.rag = None
        self.config = config or {}

    def reply(
        self,
        x: dict = None,
    ) -> dict:
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
        if x is not None:
            self.memory.add(x)
            # retrieve when the input is not None
            content = x.get("content", "")
            retrieved_docs = self.rag.retrieve(content, to_list_strs=True)
            for content in retrieved_docs:
                retrieved_docs_to_string += "\n>>>> " + content

            self.speak("[retrieved]:" + retrieved_docs_to_string)
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

        # call llm and generate response
        response = self.model(prompt).text
        msg = Msg(self.name, response)

        # Print/speak the message in this agent's voice
        self.speak(msg)

        # Record the message in memory
        self.memory.add(msg)

        return msg


class LlamaIndexAgent(RAGAgent):
    """
    A LlamaIndex agent build on LlamaIndex.
    """

    def __init__(
        self,
        name: str,
        sys_prompt: Optional[str] = None,
        model_config_name: str = None,
        emb_model_config_name: str = None,
        use_memory: bool = True,
        memory_config: Optional[dict] = None,
        prompt_type: Optional[PromptType] = PromptType.LIST,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            emb_model_config_name=emb_model_config_name,
            use_memory=use_memory,
            memory_config=memory_config,
            prompt_type=prompt_type,
            config=config,
        )
        # init rag related attributes
        self.rag = LlamaIndexRAG(
            model=self.model,
            emb_model=self.emb_model,
            config=config,
        )
        # load the document to memory
        # Feed the AgentScope tutorial documents, so that
        # the agent can answer questions related to AgentScope!
        docs = self.rag.load_data(
            loader=SimpleDirectoryReader(self.config["data_path"]),
        )
        self.rag.store_and_index(docs)


class LangChainRAGAgent(RAGAgent):
    """
    A LlamaIndex agent build on LlamaIndex.
    """

    def __init__(
        self,
        name: str,
        sys_prompt: Optional[str] = None,
        model_config_name: str = None,
        emb_model_config_name: str = None,
        use_memory: bool = True,
        memory_config: Optional[dict] = None,
        prompt_type: Optional[PromptType] = PromptType.LIST,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            emb_model_config_name=emb_model_config_name,
            use_memory=use_memory,
            memory_config=memory_config,
            prompt_type=prompt_type,
            config=config,
        )
        # init rag related attributes
        self.rag = LangChainRAG(
            model=self.model,
            emb_model=self.emb_model,
            config=config,
        )
        # load the document to memory
        # Feed the AgentScope tutorial documents, so that
        # the agent can answer questions related to AgentScope!
        docs = self.rag.load_data(
            loader=DirectoryLoader(self.config["data_path"]),
        )
        self.rag.store_and_index(docs)
