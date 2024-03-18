# -*- coding: utf-8 -*-
"""
This module is an integration of the Llama index RAG
into AgentScope package
"""

from typing import Any, Optional, List, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStore,
)
from llama_index.core.bridge.pydantic import PrivateAttr


from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)

from agentscope.rag import RAGBase
from agentscope.models import ModelWrapperBase


class _EmbeddingModel(BaseEmbedding):
    """
    wrapp a ModelWrapperBase to an embedding mode in Llama Index.
    """

    _emb_model_wrapper: ModelWrapperBase = PrivateAttr()

    def __init__(
        self,
        emb_model: ModelWrapperBase,
        embed_batch_size: int = 1,
    ) -> None:
        super().__init__(
            model_name="Temporary_embedding_wrapper",
            embed_batch_size=embed_batch_size,
        )
        self._emb_model_wrapper = emb_model

    def _get_query_embedding(self, query: str) -> List[float]:
        # Note: AgentScope embedding model wrapper returns list of embedding
        return list(self._emb_model_wrapper(query).embedding[0])

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        results = [
            list(self._emb_model_wrapper(t).embedding[0]) for t in texts
        ]
        return results

    def _get_text_embedding(self, text: str) -> Embedding:
        return list(self._emb_model_wrapper(text).embedding[0])

    # TODO: use proper async methods, but depends on model wrapper
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return self._get_text_embeddings(texts)


class LlamaIndexRAG(RAGBase):
    """
    This class is a wrapper around the Llama index RAG.
    """

    def __init__(
        self,
        model: Optional[ModelWrapperBase],
        emb_model: Union[ModelWrapperBase, BaseEmbedding],
        **kwargs: Any,
    ) -> None:
        super().__init__(model, emb_model, **kwargs)
        self.retriever = None
        self.index = None
        self.persist_dir = kwargs.get("persist_dir", "./")

    def load_data(
        self,
        loader: BaseReader = SimpleDirectoryReader,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Accept a loader, loading the desired data (no chunking)
        :param loader: object to load data, expected be an instance of class
        inheriting from BaseReader.
        :param query: optional, used when the data is in a database.
        :return: the loaded documents (un-chunked)

        Example 1: use simple directory loader to load general documents,
        including Markdown, PDFs, Word documents, PowerPoint decks, images,
        audio and video.
        ```
            load_data_to_chunks(
                loader=SimpleDirectoryReader("./data")
            )
        ```

        Example 2: use SQL loader
        ```
            load_data_to_chunks(
                DatabaseReader(
                    scheme=os.getenv("DB_SCHEME"),
                    host=os.getenv("DB_HOST"),
                    port=os.getenv("DB_PORT"),
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASS"),
                    dbname=os.getenv("DB_NAME"),
                ),
                query = "SELECT * FROM users"
            )
        ```
        """
        if query is None:
            documents = loader.load_data()
        else:
            documents = loader.load_data(query)
        return documents

    def store_and_index(
        self,
        docs: Any,
        vector_store: Union[BasePydanticVectorStore, VectorStore, None] = None,
        retriever: Optional[BaseRetriever] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Preprocessing the loaded documents.
        :param docs: documents to be processed
        :param vector_store: vector store
        :param retriever: optional, specifies the retriever to use
        :param args: additional
        :param kwargs:

        In LlamaIndex terms, an Index is a data structure composed
        of Document objects, designed to enable querying by an LLM.
        For example:
        1) preprocessing documents with
        2) generate embedding,
        3) store the embedding-content to vdb
        """
        # build and run preprocessing pipeline
        transformations = []
        if "transformations" in kwargs:
            for item in kwargs["transformations"]:
                if isinstance(item, NodeParser):
                    transformations.append(item)

        # adding embedding model as the last step of transformation
        # https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/root.html
        if isinstance(self.emb_model, ModelWrapperBase):
            transformations.append(_EmbeddingModel(self.emb_model))
        elif isinstance(self.emb_model, BaseEmbedding):
            transformations.append(self.emb_model)

        if vector_store is not None:
            pipeline = IngestionPipeline(
                transformations=transformations,
                vector_store=vector_store,
            )
            _ = pipeline.run(docs)
            self.index = VectorStoreIndex.from_vector_store(vector_store)
        else:
            # No vector store is provide, use simple in memory
            pipeline = IngestionPipeline(
                transformations=transformations,
            )
            nodes = pipeline.run(documents=docs)
            self.index = VectorStoreIndex(nodes=nodes)

        if retriever is None:
            self.retriever = self.index.as_retriever(**kwargs)
        else:
            self.retriever = retriever
        return self.index

    def set_retriever(self, retriever: BaseRetriever) -> None:
        """
        Reset the retriever if necessary.
        """
        self.retriever = retriever

    def retrieve(self, query: str) -> list[Any]:
        """
        This is a basic retrieve function
        :param query: query is expected to be a question in string

        More advanced query processing can refer to
        https://docs.llamaindex.ai/en/stable/examples/query_transformations/query_transform_cookbook.html
        """
        retrieved = self.retriever.retrieve(str(query))
        return retrieved
