# -*- coding: utf-8 -*-
"""
This module is an integration of the Llama index RAG
into AgentScope package
"""

import os.path
from typing import Any, Optional, List, Union
from loguru import logger

try:
    import llama_index
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.base.embeddings.base import (
        BaseEmbedding,
        Embedding,
    )
    from llama_index.core.ingestion import IngestionPipeline

    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core import (
        VectorStoreIndex,
        StorageContext,
        load_index_from_storage,
    )
    from llama_index.core.schema import (
        Document,
        TransformComponent,
    )
except ImportError:
    llama_index = None
    BaseRetriever = None
    BaseEmbedding = None
    Embedding = None
    IngestionPipeline = None
    SentenceSplitter = None
    VectorStoreIndex = None
    StorageContext = None
    load_index_from_storage = None
    PrivateAttr = None
    Document = None
    TransformComponent = None

from agentscope.manager import FileManager
from agentscope.models import ModelWrapperBase
from agentscope.constants import (
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from agentscope.rag.knowledge import Knowledge


try:

    class _EmbeddingModel(BaseEmbedding):
        """
        wrapper for ModelWrapperBase to an embedding model can be used
        in Llama Index pipeline.
        """

        _emb_model_wrapper: ModelWrapperBase = PrivateAttr()

        def __init__(
            self,
            emb_model: ModelWrapperBase,
            embed_batch_size: int = 1,
        ) -> None:
            """
            Dummy wrapper to convert a ModelWrapperBase to llama Index
            embedding model

            Args:
                emb_model (ModelWrapperBase):
                    embedding model in ModelWrapperBase
                embed_batch_size (int):
                    batch size, defaults to 1
            """
            super().__init__(
                model_name="Temporary_embedding_wrapper",
                embed_batch_size=embed_batch_size,
            )
            self._emb_model_wrapper = emb_model

        def _get_query_embedding(self, query: str) -> List[float]:
            """
            get embedding for query
            Args:
                query (str): query to be embedded
            """
            # Note: AgentScope embedding model wrapper returns list
            # of embedding
            return list(self._emb_model_wrapper(query).embedding[0])

        def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
            """
            get embedding for list of strings
            Args:
                 texts ( List[str]): texts to be embedded
            """
            results = [
                list(self._emb_model_wrapper(t).embedding[0]) for t in texts
            ]
            return results

        def _get_text_embedding(self, text: str) -> Embedding:
            """
            get embedding for a single string
            Args:
                 text (str): texts to be embedded
            """
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

except Exception:

    class _EmbeddingModel:  # type: ignore[no-redef]
        """
        A dummy embedding model for passing tests when
        llama-index is not install
        """

        def __init__(self, emb_model: ModelWrapperBase):
            self._emb_model_wrapper = emb_model


class LlamaIndexKnowledge(Knowledge):
    """
    This class is a wrapper with the llama index RAG.
    """

    def __init__(
        self,
        knowledge_id: str,
        emb_model: Union[ModelWrapperBase, BaseEmbedding, None] = None,
        knowledge_config: Optional[dict] = None,
        model: Optional[ModelWrapperBase] = None,
        persist_root: Optional[str] = None,
        overwrite_index: Optional[bool] = False,
        showprogress: Optional[bool] = True,
        **kwargs: Any,
    ) -> None:
        """
        initialize the knowledge component based on the
        llama-index framework: https://github.com/run-llama/llama_index

        Notes:
            In LlamaIndex, one of the most important concepts is index,
            which is a data structure composed of Document objects, designed to
            enable querying by an LLM. The core workflow of initializing RAG is
            to convert data to index, and retrieve information from index.
            For example:
            1) preprocessing documents with data loaders
            2) generate embedding by configuring pipline with embedding models
            3) store the embedding-content to vector database
                the default dir is "./rag_storage/knowledge_id"

        Args:
            knowledge_id (str):
                The id of the RAG knowledge unit.
            emb_model (ModelWrapperBase):
                The embedding model used for generate embeddings
            knowledge_config (dict):
                The configuration for llama-index to
                generate or load the index.
            model (ModelWrapperBase):
                The language model used for final synthesis
            persist_root (str):
                The root directory for index persisting
            overwrite_index (Optional[bool]):
                Whether to overwrite the index while refreshing
            showprogress (Optional[bool]):
                Whether to show the indexing progress
        """
        super().__init__(
            knowledge_id=knowledge_id,
            emb_model=emb_model,
            knowledge_config=knowledge_config,
            model=model,
            **kwargs,
        )
        if llama_index is None:
            raise ImportError(
                "LlamaIndexKnowledge require llama-index installed. "
                "Try a stable llama-index version, such as "
                "`pip install llama-index==0.10.30`",
            )

        if persist_root is None:
            persist_root = FileManager.get_instance().cache_dir or "./"
        self.persist_dir = os.path.join(persist_root, knowledge_id)
        logger.info(f"** persist_dir: {self.persist_dir}")
        self.emb_model = emb_model
        self.overwrite_index = overwrite_index
        self.showprogress = showprogress
        self.index = None
        # ensure the emb_model is compatible with LlamaIndex
        if isinstance(emb_model, ModelWrapperBase):
            self.emb_model = _EmbeddingModel(emb_model)
        elif isinstance(self.emb_model, BaseEmbedding):
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
            self._load_index()
            # self.refresh_index()
        else:
            self._data_to_index()
        self._get_retriever()
        logger.info(
            f"RAG with knowledge ids: {self.knowledge_id} "
            f"initialization completed!\n",
        )

    def _load_index(self) -> None:
        """
        Load the persisted index from persist_dir.
        """
        # load the storage_context
        # Inject the persist_dir setting
        self.knowledge_config.get("store_and_index", {}).get(
            "storage_context",
            {},
        ).update(
            {
                "persist_dir": self.persist_dir,
            },
        )
        storage_context = self._set_store(self.knowledge_config)
        if not storage_context:
            # if storage_context is not set, use the default
            storage_context = StorageContext.from_defaults(
                persist_dir=self.persist_dir,
            )
        # construct index from
        self.index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=self.emb_model,
        )
        logger.info(f"index loaded from {self.persist_dir}")

    def _data_to_index(self) -> None:
        """
        Convert the data to index by configs. This includes:
            * load the data to documents by using information from configs
            * set the transformations associated with documents
            * convert the documents to nodes
            * convert the nodes to index

        Notes:
            As each selected file type may need to use a different loader
            and transformations, knowledge_config is a list of configs.
        """
        nodes = []
        # load data to documents and set transformations
        # using information in knowledge_config
        for config in self.knowledge_config.get("data_processing"):
            documents = self._data_to_docs(config=config)
            transformations = self._set_transformations(config=config).get(
                "transformations",
            )
            nodes_docs = self._docs_to_nodes(
                documents=documents,
                transformations=transformations,
            )
            nodes = nodes + nodes_docs
        # set store
        storage_context = self._set_store(self.knowledge_config)
        # convert nodes to index
        # if storage_context is None, use the default
        self.index = VectorStoreIndex(
            nodes=nodes,
            embed_model=self.emb_model,
            storage_context=storage_context,
        )
        logger.info("index calculation completed.")
        # persist the calculated index
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        logger.info("index persisted.")

    def _data_to_docs(
        self,
        query: Optional[str] = None,
        config: dict = None,
    ) -> Any:
        """
        This method set the loader as needed, or just use the default setting.
        Then use the loader to load data from dir to documents.

        Notes:
            We can use simple directory loader (SimpleDirectoryReader)
            to load general documents, including Markdown, PDFs,
            Word documents, PowerPoint decks, images, audio and video.
            Or use SQL loader (DatabaseReader) to load database.

        Args:
            query (Optional[str]):
                optional, used when the data is in a database.
            config (dict):
                optional, used when the loader config is in a config file.
        Returns:
            Any: loaded documents
        """
        loader = self._set_loader(config=config).get("loader")
        # let the doc_id be the filename for each document
        loader.filename_as_id = True
        if query is None:
            documents = loader.load_data()
        else:
            # this is for querying a database,
            # does not work for loading a document directory
            documents = loader.load_data(query)
        logger.info(f"loaded {len(documents)} documents")
        return documents

    def _docs_to_nodes(
        self,
        documents: List[Document],
        transformations: Optional[list[Optional[TransformComponent]]] = None,
    ) -> Any:
        """
        Convert the loaded documents to nodes using transformations.

        Args:
            documents (List[Document]):
                documents to be processed, usually expected to be in
                 llama index Documents.
            transformations (Optional[list[TransformComponent]]):
                optional, specifies the transformations (operators) to
                process documents (e.g., split the documents into smaller
                chunks)
        Return:
            Any: return the index of the processed document
        """
        # nodes, or called chunks, is a presentation of the documents
        # we build nodes by using the IngestionPipeline
        # for each document with corresponding transformations
        pipeline = IngestionPipeline(
            transformations=transformations,
        )
        # stack up the nodes from the pipline
        nodes = pipeline.run(
            documents=documents,
            show_progress=self.showprogress,
        )
        logger.info("nodes generated.")
        return nodes

    def _set_loader(self, config: dict) -> Any:
        """
        Set the loader as needed, or just use the default setting.

        Args:
            config (dict): a dictionary containing configurations
        """
        if "load_data" in config:
            # we prepare the loader from the configs
            loader = self._prepare_args_from_config(
                config=config.get("load_data", {}),
            )
        else:
            # we prepare the loader by default
            try:
                from llama_index.core import SimpleDirectoryReader
            except ImportError as exc_inner:
                raise ImportError(
                    " LlamaIndexAgent requires llama-index to be install."
                    "Please run `pip install llama-index`",
                ) from exc_inner
            loader = {
                "loader": SimpleDirectoryReader(
                    input_dir="set_default_data_path",
                ),
            }
        logger.info("loaders are ready.")
        return loader

    def _set_transformations(self, config: dict) -> Any:
        """
        Set the transformations as needed, or just use the default setting.

        Args:
            config (dict): a dictionary containing configurations.
        """
        if "data_parse" in config:
            temp = self._prepare_args_from_config(
                config=config.get("data_parse", {}),
            )
            transformations = temp.get("transformations")
        elif "store_and_index" in config:
            logger.warning(
                "The old configuration structure is deprecated, "
                "please use data_parse instead of store_and_index.",
            )
            temp = self._prepare_args_from_config(
                config=config.get("store_and_index", {}),
            )
            transformations = temp.get("transformations")
        else:
            transformations = [
                SentenceSplitter(
                    chunk_size=self.knowledge_config.get(
                        "chunk_size",
                        DEFAULT_CHUNK_SIZE,
                    ),
                    chunk_overlap=self.knowledge_config.get(
                        "chunk_overlap",
                        DEFAULT_CHUNK_OVERLAP,
                    ),
                ),
            ]
        # adding embedding model as the last step of transformation
        # https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/root.html
        transformations.append(self.emb_model)
        logger.info("transformations are ready.")
        # as the last step, we need to repackage the transformations in dict
        transformations = {"transformations": transformations}
        return transformations

    def _set_store(self, config: dict) -> Any:
        """
        Set the store as needed, or just use the default setting.

        Args:
            config (dict): a dictionary containing configurations.
        """
        if "store_and_index" in config:
            temp = self._prepare_args_from_config(
                config=config.get("store_and_index", {}),
            )
            context_config = temp.get("storage_context")
        else:
            return None

        # Create the storage context
        storage_context = StorageContext.from_defaults(**context_config)
        logger.info("storage_context is ready.")
        return storage_context

    def _get_retriever(
        self,
        similarity_top_k: int = None,
        **kwargs: Any,
    ) -> BaseRetriever:
        """
        Set the retriever as needed, or just use the default setting.

        Args:
            retriever (Optional[BaseRetriever]): passing a retriever in
             LlamaIndexKnowledge
            rag_config (dict): rag configuration, including similarity top k
            index.
        """
        # set the retriever
        logger.info(
            f"similarity_top_k" f"={similarity_top_k or DEFAULT_TOP_K}",
        )
        retriever = self.index.as_retriever(
            embed_model=self.emb_model,
            similarity_top_k=similarity_top_k or DEFAULT_TOP_K,
            **kwargs,
        )
        logger.info("retriever is ready.")
        return retriever

    def retrieve(
        self,
        query: str,
        similarity_top_k: int = None,
        to_list_strs: bool = False,
        retriever: Optional[BaseRetriever] = None,
        **kwargs: Any,
    ) -> list[Any]:
        """
        This is a basic retrieve function for knowledge.
        It will build a retriever on the fly and return the
        result of the query.
        Args:
            query (str):
                query is expected to be a question in string
            similarity_top_k (int):
                the number of most similar data returned by the
                retriever.
            to_list_strs (bool):
                whether returns the list of strings;
                if False, return NodeWithScore
            retriever (BaseRetriever):
                for advanced usage, user can pass their own retriever.
        Return:
            list[Any]: list of str or NodeWithScore

        More advanced query processing can refer to
        https://docs.llamaindex.ai/en/stable/examples/query_transformations/query_transform_cookbook.html
        """
        if retriever is None:
            retriever = self._get_retriever(similarity_top_k)
        retrieved = retriever.retrieve(str(query))
        if to_list_strs:
            results = []
            for node in retrieved:
                results.append(node.get_text())
            return results
        return retrieved

    def refresh_index(self) -> None:
        """
        Refresh the index when needed.
        """
        for config in self.knowledge_config.get("data_processing"):
            documents = self._data_to_docs(config=config)
            # store and indexing for each file type
            transformations = self._set_transformations(config=config).get(
                "transformations",
            )
            self._insert_docs_to_index(
                documents=documents,
                transformations=transformations,
            )

    def _insert_docs_to_index(
        self,
        documents: List[Document],
        transformations: TransformComponent,
    ) -> None:
        """
        Add documents to the index. Given a list of documents, we first test if
        the doc_id is already in the index. If not, we add the doc to the
        list. If yes, and the over-write flag is enabled,
        we delete the old doc and add the new doc to the list.
        Lastly, we generate nodes for all documents on the list, and insert
        the nodes to the index.

        Args:
            documents (List[Document]): list of documents to be added.
            transformations (TransformComponent): transformations that
            convert the documents into nodes.
        """
        # this is the pipline that generate the nodes
        pipeline = IngestionPipeline(
            transformations=transformations,
        )
        # we need to generate nodes from this list of documents
        insert_docs_list = []
        for doc in documents:
            if doc.doc_id not in self.index.ref_doc_info.keys():
                # if the doc_id is not in the index, we add it to the list
                insert_docs_list.append(doc)
                logger.info(
                    f"add new documents to index, " f"doc_id={doc.doc_id}",
                )
            else:
                if self.overwrite_index:
                    # if we enable overwrite index, we delete the old doc
                    self.index.delete_ref_doc(
                        ref_doc_id=doc.doc_id,
                        delete_from_docstore=True,
                    )
                    # then add the same doc to the list
                    insert_docs_list.append(doc)
                    logger.info(
                        f"replace document in index, " f"doc_id={doc.doc_id}",
                    )
        logger.info("documents scan completed.")
        # we generate nodes for documents on the list
        nodes = pipeline.run(
            documents=insert_docs_list,
            show_progress=True,
        )
        logger.info("nodes generated.")
        # insert the new nodes to index
        self.index.insert_nodes(nodes=nodes)
        logger.info("nodes inserted to index.")
        # persist the updated index
        self.index.storage_context.persist(persist_dir=self.persist_dir)

    def _delete_docs_from_index(
        self,
        documents: List[Document],
    ) -> None:
        """
        Delete the nodes that are associated with a list of documents.

        Args:
            documents (List[Document]): list of documents to be deleted.
        """
        doc_id_list = [doc.doc_id for doc in documents]
        for key in self.index.ref_doc_info.keys():
            if key in doc_id_list:
                self.index.delete_ref_doc(
                    ref_doc_id=key,
                    delete_from_docstore=True,
                )
                logger.info(f"docs deleted from index, doc_id={key}")
        # persist the updated index
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        logger.info("nodes delete completed.")
