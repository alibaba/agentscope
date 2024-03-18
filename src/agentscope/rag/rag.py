# -*- coding: utf-8 -*-
"""
Base class module for retrieval augmented generation (RAG).
To accommodate the RAG process of different packages,
we abstract the RAG process into four stages:
- data loading: loading data into memory for following processing;
- data indexing and storage: document chunking, embedding generation,
and off-load the data into VDB;
- data retrieval: taking a query and return a batch of documents or
document chunks;
- post-processing of the retrieved data: use the retrieved data to
generate an answer.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from agentscope.models import ModelWrapperBase


class RAGBase(ABC):
    """Base class for RAG"""

    def __init__(
        self,
        model: Optional[ModelWrapperBase],
        emb_model: Optional[ModelWrapperBase],
        **kwargs: Any,
    ) -> None:
        # pylint: disable=unused-argument
        self.postprocessing_model = model
        self.emb_model = emb_model

    @abstractmethod
    def load_data(
        self,
        loader: Any,
        query: Any,
        **kwargs: Any,
    ) -> Any:
        """
        load data (documents) from disk to memory and chunking them
        """

    @abstractmethod
    def store_and_index(
        self,
        docs: Any,
        vector_store: Any,
        **kwargs: Any,
    ) -> Any:
        """
        preprocessing the loaded documents, for example:
        1) chunking,
        2) generate embedding,
        3) store the embedding-content to vdb
        """

    @abstractmethod
    def retrieve(self, query: Any) -> list[Any]:
        """
        retrieve list of content from vdb to memory
        """

    def post_processing(
        self,
        retrieved_docs: list[Any],
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        """
        post-processing function, generates answer based on the
        retrieved documents.
        Example:
            self.postprocessing_model(prompt.format(retrieved_docs))
        """
        raise NotImplementedError
