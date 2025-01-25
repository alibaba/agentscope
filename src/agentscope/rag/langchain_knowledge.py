# -*- coding: utf-8 -*-
"""
This module is an integration of the Langchain RAG
into AgentScope package
"""
import os
import json
from typing import Any, Optional, List, Union
from loguru import logger
from pydantic import BaseModel

try:
    import langchain
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.vectorstores import (
        InMemoryVectorStore,
        VectorStoreRetriever,
    )
    from langchain_core.embeddings import Embeddings
    from langchain_core.indexing import InMemoryRecordManager
    from langchain.indexes import index
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_text_splitters.base import TextSplitter

except Exception:
    langchain = None
    Document = None
    BaseRetriever = None
    TextSplitter = None
    VectorStoreRetriever = None
    InMemoryVectorStore = None
    InMemoryRecordManager = None
    index = None
    Embeddings = None
    CharacterTextSplitter = None

from agentscope.manager import FileManager
from agentscope.models import ModelWrapperBase
from agentscope.constants import (
    DEFAULT_TOP_K,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from agentscope.rag.knowledge import Knowledge

try:

    class _EmbeddingModel(BaseModel, Embeddings):
        _emb_model_wrapper: ModelWrapperBase

        def __init__(self, emb_model: ModelWrapperBase, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._emb_model_wrapper = emb_model

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """
            embed a list of strings
            Args:
                texts (List[str]): texts to be embedded
            """
            results = [
                list(self._emb_model_wrapper(t).embedding[0]) for t in texts
            ]
            return results

        def embed_query(self, text: str) -> List[float]:
            """
            embeds a single query text into a vector representation
            Args:
                text (str): The query text to embed.
            """
            return self.embed_documents([text])[0]

except Exception:

    class _EmbeddingModel:  # type: ignore[no-redef]
        """
        A dummy embedding model for passing tests when
        langchain is not installed
        """

        def __init__(self, emb_model: ModelWrapperBase):
            self._emb_model_wrapper = emb_model


class LangChainKnowledge(Knowledge):
    """
    This class is a wrapper with the langchain RAG.
    """

    def __init__(
        self,
        knowledge_id: str,
        emb_model: Union[ModelWrapperBase, Embeddings, None] = None,
        knowledge_config: Optional[dict] = None,
        model: Optional[ModelWrapperBase] = None,
        persist_root: Optional[str] = None,
        overwrite_index: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            knowledge_id=knowledge_id,
            emb_model=emb_model,
            knowledge_config=knowledge_config,
            model=model,
            **kwargs,
        )

        if langchain is None:
            raise ImportError(
                "Please install langchain first.",
            )

        if persist_root is None:
            persist_root = FileManager.get_instance().cache_dir or "./"
        self.persist_dir = os.path.join(persist_root, knowledge_id)
        self.persist_store_file = os.path.join(
            self.persist_dir,
            "vector_store.json",
        )
        self.persist_index_file = os.path.join(self.persist_dir, "index.json")
        self.emb_model = emb_model
        self.overwrite_index = overwrite_index
        self.vectorstore = None
        self.record_manager = None
        # ensure the emb_model is compatible with Langchian
        if isinstance(emb_model, ModelWrapperBase):
            self.emb_model = _EmbeddingModel(emb_model)
        elif isinstance(self.emb_model, Embeddings):
            pass
        else:
            raise TypeError(
                f"Embedding model does not support {type(self.emb_model)}.",
            )
        # then we can initialize the RAG
        self._init_rag()

    def _init_rag(self, **kwargs: Any) -> None:
        """
        Initialize the RAG. This includes:
            * if the persist_dir exists, load the persisted index
            * if not, convert the data to index
            * if needed, update the index
            * set the retriever to retrieve information from index

        Notes:
            * the index is persisted in the self.persist_dir
            * the refresh_index method is placed here for testing, it can be
                called externally. For example, updated the index periodically
                by calling rag.refresh_index() during the execution of the
                agent.
        """
        if os.path.exists(self.persist_dir):
            self._load_store()
            # self.refresh_index()
        else:
            self._data_to_store()
        logger.info(
            f"RAG with knowledge ids: {self.knowledge_id} "
            f"initialization completed!\n",
        )

    def _load_store(self) -> None:
        """
        Load the persisted index from persist_dir.
        """
        # set the storage
        self.vectorstore = self._set_store(
            self.knowledge_config.get("store_and_index", {}),
        )
        if not self.vectorstore:
            self.vectorstore = InMemoryVectorStore.load(
                self.persist_store_file,
                self.emb_model,
            )
        # set the record manager
        self.record_manager = InMemoryRecordManager(self.knowledge_id)
        self.record_manager.create_schema()
        self._load_memory_record(self.persist_index_file)
        logger.info(f"vector store and index loaded from {self.persist_dir}")

    def _data_to_store(self) -> None:
        # create the record manager
        self.record_manager = InMemoryRecordManager(self.knowledge_id)
        self.record_manager.create_schema()

        chunks = []
        for config in self.knowledge_config.get("data_processing"):
            documents = self._data_to_docs(config=config)
            splitter = self._set_splitter(config=config).get("splitter")
            chunks_docs = self._docs_to_chunks(
                documents=documents,
                splitter=splitter,
            )
            chunks = chunks + chunks_docs

        # convert chunks to vector store and index
        self.vectorstore = self._set_store(
            config=self.knowledge_config.get("store_and_index", {}),
        )
        if not self.vectorstore:
            self.vectorstore = InMemoryVectorStore(
                self.emb_model,
            )
        index(
            chunks,
            self.record_manager,
            self.vectorstore,
            cleanup=None,
            source_id_key="source",
            # upsert_kwargs={"embedding": self.emb_model}
            # This feature is only supported in langchain 0.3.10
        )
        logger.info("vector store and index created successfully.")

        # persist
        if isinstance(self.vectorstore, InMemoryVectorStore):
            self.vectorstore.dump(self.persist_store_file)
            logger.info("In-memory vector store are persisted.")
        self._save_memory_record(self.persist_index_file)
        logger.info("index are persisted.")

    def _save_memory_record(self, filename: str) -> None:
        filedir = os.path.dirname(filename)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.record_manager.records, f, indent=4)

    def _load_memory_record(self, filename: str) -> None:
        with open(filename, "r", encoding="utf-8") as f:
            self.record_manager.records = json.load(f)

    def _data_to_docs(
        self,
        config: dict = None,
    ) -> Any:
        """
        This method set the loader as needed, or just use the default setting.
        Then use the loader to load data from dir to documents.

        Notes:
            We can use directory loader (DirectoryReader)
            to load general documents, including Markdown, PDFs,
            Word documents, PowerPoint decks, images, audio and video.

        Args:
            config (dict):
                optional, used when the loader config is in a config file.
        Returns:
            Any: loaded documents
        """
        loader = self._set_loader(config=config).get("loader")
        documents = loader.load()
        logger.info(f"loaded {len(documents)} documents")
        return documents

    def _docs_to_chunks(
        self,
        documents: List[Document],
        splitter: Optional[TextSplitter],
    ) -> Any:
        return splitter.split_documents(documents)

    def _set_store(self, config: dict) -> Any:
        if "stores" in config:
            init_config = (
                config.get("stores", {})
                .get("vector_store", {})
                .get("init_args", {})
            )
            # we prepare the ~embedding_key from the configs
            embedding_key = init_config.pop(
                "embedding_key",
                "embedding",
            )
            init_config[embedding_key] = self.emb_model
            temp = self._prepare_args_from_config(
                config=config.get("stores", {}),
            )
            vector_store = temp.get("vector_store")
        else:
            vector_store = None
        return vector_store

    def _set_loader(self, config: dict) -> Any:
        """
        Set the loader as needed, or just use the default setting.

        Args:
            config (dict): a dictionary containing configurations
        """
        if "load_data" in config:
            # we prepare the ~loader from the configs
            loader = self._prepare_args_from_config(
                config=config.get("load_data", {}),
            )
        else:
            # we prepare the loader by default
            try:
                from langchain_community.document_loaders import (
                    DirectoryLoader,
                )
            except ImportError as exc_inner:
                raise ImportError(
                    " LangChainAgent requires langchain to be install."
                    "Please run `pip install langchain`",
                ) from exc_inner
            loader = {
                "loader": DirectoryLoader(
                    path="set_default_data_path",
                ),
            }
        logger.info("loaders are ready.")
        return loader

    def _set_splitter(self, config: dict) -> Any:
        """
        Set the splitter as needed, or just use the default setting.

        Args:
            config (dict): a dictionary containing configurations.
        """
        if "data_parse" in config:
            temp = self._prepare_args_from_config(
                config=config.get("data_parse", {}),
            )
            splitter = temp.get("splitter")
        elif "store_and_index" in config:
            logger.warning(
                "The old configuration structure is deprecated, "
                "please use data_parse instead of store_and_index.",
            )
            temp = self._prepare_args_from_config(
                config=config.get("store_and_index", {}),
            )
            splitter = temp.get("splitter")
        else:
            splitter = CharacterTextSplitter(
                chunk_size=self.knowledge_config.get(
                    "chunk_size",
                    DEFAULT_CHUNK_SIZE,
                ),
                chunk_overlap=self.knowledge_config.get(
                    "chunk_overlap",
                    DEFAULT_CHUNK_OVERLAP,
                ),
            )
        logger.info("splitter are ready.")
        splitter = {"splitter": splitter}
        return splitter

    def _get_retriever(
        self,
        search_type: str,
        search_kwargs: dict,
    ) -> BaseRetriever:
        # set the retriever
        default_kwargs = {
            "k": DEFAULT_TOP_K,
            "score_threshold": (
                DEFAULT_SCORE_THRESHOLD
                if search_type == "similarity_score_threshold"
                else None
            ),
        }
        default_kwargs = {
            key: value
            for key, value in default_kwargs.items()
            if value is not None
        }
        search_kwargs = {**default_kwargs, **search_kwargs}
        logger.info(
            f"search_type: {search_type}; search_kwargs: {search_kwargs}",
        )
        retriever = VectorStoreRetriever(
            vectorstore=self.vectorstore,
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
        logger.info("retriever is ready.")
        return retriever

    def retrieve(
        self,
        query: str,
        similarity_top_k: int = None,
        to_list_strs: bool = False,
        search_type: str = "similarity",
        search_kwargs: dict = None,
        retriever: Optional[BaseRetriever] = None,
        **kwargs: Any,
    ) -> list[Any]:
        search_kwargs = search_kwargs or {}
        if similarity_top_k:
            search_kwargs.update({"k": similarity_top_k})
        if retriever is None:
            retriever = self._get_retriever(search_type, search_kwargs)
        retrieved = retriever.invoke(str(query))
        if to_list_strs:
            results = []
            for node in retrieved:
                results.append(node.page_content)
            return results
        return retrieved

    def refresh_index(self) -> None:
        """
        Refresh the index when needed.
        """
        clean_results = []
        for config in self.knowledge_config.get("data_processing"):
            documents = self._data_to_docs(config=config)
            splitter = self._set_splitter(config=config).get(
                "splitter",
            )
            chunks_docs = self._docs_to_chunks(
                documents=documents,
                splitter=splitter,
            )
            if self.overwrite_index:
                clean_method = "incremental"
            else:
                clean_method = None
            clean_result = index(
                chunks_docs,
                self.record_manager,
                self.vectorstore,
                cleanup=clean_method,
                source_id_key="source",
            )
            clean_results.append(clean_result)

        logger.info(f"Refresh result: {clean_results}")
