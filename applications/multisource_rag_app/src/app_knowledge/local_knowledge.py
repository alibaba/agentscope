# -*- coding: utf-8 -*-
# pylint: disable=W0212
"""
Knowledge modules specialized for RAG application
"""
import os
import pickle
from typing import List, Union, Any, Sequence, Optional

import numpy as np
from llama_index.core import (
    VectorStoreIndex,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import (
    BaseNode,
    TextNode,
)
from llama_index.core.vector_stores.types import VectorStore
from loguru import logger
from scipy.cluster.hierarchy import fclusterdata

from agentscope.rag.llama_index_knowledge import (
    LlamaIndexKnowledge,
    DEFAULT_TOP_K,
)


class LocalKnowledge(LlamaIndexKnowledge):
    """
    Specialized Knowledge class for the application,
    including embedding routing
    """

    knowledge_type = "local_knowledge"

    def __init__(self, **kwargs: Any):
        self.bm25_retriever = None
        self.cluster_means = None
        self.language = None
        super().__init__(**kwargs)

    def _data_to_index(
        self,
        vector_store: Optional[VectorStore] = None,
    ) -> List[BaseNode]:
        """
        Overwrite with clustering_by_embedding
        """
        nodes = super()._data_to_index(vector_store)
        self.clustering_by_embedding(nodes)
        return nodes

    def clustering_by_embedding(
        self,
        nodes: List[BaseNode],
        clust_num: int = 128,
        update: bool = False,
    ) -> Sequence:
        """
        Clustering the embeddings
        """
        if update:
            with open(self._get_embeddings_filepath(), "rb") as f:
                existing_embeddings = pickle.load(f)
            embeddings = existing_embeddings + [n.embedding for n in nodes]
        else:
            embeddings = [n.embedding for n in nodes]

        logger.info(
            "update all embedding and store to "
            f"{self._get_embeddings_filepath()}",
        )
        with open(self._get_embeddings_filepath(), "wb") as f:
            pickle.dump(embeddings, f)

        clust_num = min(clust_num, len(nodes))
        logger.info(
            "clustering nodes based on embeddings into "
            f"{clust_num} clusters....",
        )

        clustering_result = fclusterdata(
            embeddings,
            t=clust_num,
            criterion="maxclust",
            metric="cosine",
            method="average",
        )
        clusters = {}
        for cluster_idx, emb in zip(clustering_result, embeddings):
            if cluster_idx not in clusters:
                clusters[cluster_idx] = []
            clusters[cluster_idx].append(emb)

        cluster_means = []
        for _, item in clusters.items():
            cluster_means.append(np.mean(item, axis=0))

        cluster_means = np.array(cluster_means)

        # save clustering means
        self._save_clustering_means(cluster_means)

        self.cluster_means = cluster_means
        logger.info("clustering nodes and compute means done.")

        return cluster_means

    def _get_embeddings_filepath(self) -> str:
        return os.path.join(
            self.persist_dir,
            self.knowledge_id + "_all_embeddings.pkl",
        )

    def _get_clustering_mean_filepath(self) -> str:
        return os.path.join(
            self.persist_dir,
            self.knowledge_id + "_cluster_means.pkl",
        )

    def _save_clustering_means(self, cluster_means: Sequence) -> None:
        file_name = self._get_clustering_mean_filepath()
        with open(file_name, "wb") as f:
            pickle.dump(cluster_means, f)


class ESKnowledge(LocalKnowledge):
    """
    This a specialized subclass of LlamaIndex Knowledge using Elasticsearch
    """

    knowledge_type = "es_knowledge"
    async_client = None

    def _init_rag(self) -> None:
        if "es_config" in self.knowledge_config:
            self._setup_with_es()
        else:
            raise ValueError("ESKnowledge require ES config")

        self._get_retriever()
        logger.info(
            f"RAG with knowledge ids: {self.knowledge_id} "
            f"initialization completed!\n",
        )

    def _setup_with_es(self) -> None:
        from elasticsearch import AsyncElasticsearch, Elasticsearch
        from llama_index.vector_stores.elasticsearch import ElasticsearchStore

        es_config = self.knowledge_config.get("es_config", {})
        mode = es_config.get("mode", "read-only")

        self.async_client = AsyncElasticsearch(
            hosts=[es_config["vector_store_args"]["es_url"]],
            node_class="httpxasync",  # use httpx instead of aiohttp
        )
        sync_client = Elasticsearch(
            hosts=[es_config["vector_store_args"]["es_url"]],
        )
        if not sync_client.indices.exists(index=self.knowledge_id):
            sync_client.indices.create(index=self.knowledge_id)
        vector_store = ElasticsearchStore(
            index_name=self.knowledge_id,
            es_url=es_config["vector_store_args"]["es_url"],
            es_client=self.async_client,
        )
        assert vector_store is not None
        if mode == "read-only":
            logger.info(
                f"[READ-ONLY MODE] Connect VDB for {self.knowledge_id}",
            )
            # storage_context = StorageContext.from_defaults(vector_store=es)
            self.index = VectorStoreIndex.from_vector_store(
                vector_store,
                self.emb_model,
            )
        elif mode in ["update"]:
            logger.info(
                f"[UPDATE MODE] update {self.knowledge_id} in VDB with data"
                "with data processing config "
                f"{self.knowledge_config.get('data_processing')}",
            )
            self._data_to_index(vector_store)

        logger.info("[Done] Index with VDB")

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
        from llama_index.vector_stores.elasticsearch import (
            ElasticsearchStore,
            AsyncBM25Strategy,
        )

        # set the retriever
        logger.info(
            f"similarity_top_k" f"={similarity_top_k or DEFAULT_TOP_K}",
        )
        retriever = self.index.as_retriever(
            embed_model=self.emb_model,
            similarity_top_k=similarity_top_k or DEFAULT_TOP_K,
            **kwargs,
        )

        if self.bm25_retriever is None and self.additional_sparse_retrieval:
            es_config = self.knowledge_config.get("es_config", {})
            text_store = ElasticsearchStore(
                retrieval_strategy=AsyncBM25Strategy(),
                index_name=self.knowledge_id,
                es_url=es_config["vector_store_args"]["es_url"],
                es_client=self.async_client,
                # **self.knowledge_config["es_config"]["vector_store_args"]
            )
            text_index = VectorStoreIndex.from_vector_store(
                vector_store=text_store,
                embed_model=self.emb_model,
            )
            self.bm25_retriever = text_index.as_retriever(
                similarity_top_k=similarity_top_k,
            )
        elif self.additional_sparse_retrieval:
            self.bm25_retriever.similarity_top_k = similarity_top_k

        logger.info("retriever is ready.")

        return retriever

    def _es_get_all_embeddings(self) -> Sequence:
        from elasticsearch import Elasticsearch

        es_config = self.knowledge_config.get("es_config", {})
        embedding_field = (
            "embedding"  # Replace with the actual field name for embeddings
        )
        # Initialize an empty list to store all embeddings
        all_embeddings = []
        es = Elasticsearch(
            hosts=[es_config["vector_store_args"]["es_url"]],
        )
        response = es.search(
            index=self.knowledge_id,
            body={
                "_source": [
                    embedding_field,
                ],  # Only retrieve the embedding field
                "query": {
                    "match_all": {},
                },
                "size": 1000,  # Number of documents to retrieve per request
            },
            scroll="2m",  # Keep the search context open for 2 minutes
        )
        # Get the scroll ID and hits
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]

        # Collect embeddings from the first batch of hits
        for hit in hits:
            all_embeddings.append(hit["_source"][embedding_field])

        # Continue scrolling to get all embeddings
        while len(hits) > 0:
            response = es.scroll(scroll_id=scroll_id, scroll="2m")
            hits = response["hits"]["hits"]

            for hit in hits:
                all_embeddings.append(hit["_source"][embedding_field])
        return all_embeddings

    def _cluster_means_idx_name(self) -> str:
        return self.knowledge_id + "_cluster_means"

    def _save_clustering_means(self, cluster_means: Sequence) -> None:
        from elasticsearch import Elasticsearch
        from llama_index.vector_stores.elasticsearch import ElasticsearchStore

        cluster_idx_name = self._cluster_means_idx_name()
        es_config = self.knowledge_config.get("es_config", {})
        sync_client = Elasticsearch(
            hosts=[es_config["vector_store_args"]["es_url"]],
        )
        if not sync_client.indices.exists(index=cluster_idx_name):
            sync_client.indices.create(index=cluster_idx_name)
        vector_store = ElasticsearchStore(
            index_name=cluster_idx_name,
            es_url=es_config["vector_store_args"]["es_url"],
            es_client=self.async_client,
        )
        mean_nodes = [
            TextNode(
                embedding=list(mean),
                text=self.knowledge_id,
            )
            for mean in cluster_means
        ]
        vector_store.add(mean_nodes)

    def cluster_similarities(
        self,
        query_or_embedding: Union[str, Sequence[float]],
    ) -> Sequence:
        """
        Clustering based on the embedding similarities.
        """
        from elasticsearch import Elasticsearch

        if isinstance(query_or_embedding, str):
            query_embedding = self.emb_model._get_query_embedding(
                query_or_embedding,
            )
        else:
            query_embedding = query_or_embedding

        cluster_idx_name = self._cluster_means_idx_name()
        es_config = self.knowledge_config.get("es_config", {})
        sync_client = Elasticsearch(
            hosts=[es_config["vector_store_args"]["es_url"]],
        )
        # Perform the similarity search
        response = sync_client.search(
            index=cluster_idx_name,
            body={
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": 5,  # Number of nearest neighbors to return
                    "num_candidates": 1000,  # Number of candidate vectors
                },
            },
        )
        similarity = []
        for hit in response["hits"]["hits"]:
            similarity.append(hit["_score"])

        return similarity

    def clean_es(self) -> None:
        """
        Remove the knowledge from the ElasticSearch
        """
        from elasticsearch import Elasticsearch

        es_config = self.knowledge_config.get("es_config", {})
        es = Elasticsearch(
            hosts=[es_config["vector_store_args"]["es_url"]],
        )

        index_name = self.knowledge_id
        response = es.indices.delete(index=index_name, ignore=[400, 404])

        # Check response
        if "acknowledged" in response and response["acknowledged"]:
            logger.info(f"Index {index_name} deleted successfully.")
        else:
            logger.info(f"Failed to delete index {index_name}.")

        index_cluster_name = self._cluster_means_idx_name()
        response = es.indices.delete(
            index=index_cluster_name,
            ignore=[400, 404],
        )

        # Check response
        if "acknowledged" in response and response["acknowledged"]:
            logger.info(f"Index {index_name} deleted successfully.")
        else:
            logger.info(f"Failed to delete index {index_name}.")

    def delete_files(self, file_paths: Union[str, list]) -> None:
        """
        Delete all chunks of a file
        """
        from elasticsearch import Elasticsearch

        es_config = self.knowledge_config.get("es_config", {})
        es = Elasticsearch(
            hosts=[es_config["vector_store_args"]["es_url"]],
        )
        for path in file_paths:
            response = es.delete_by_query(
                index=self.knowledge_id,
                body={
                    "query": {
                        "match": {
                            "metadata.file_path": path,
                        },
                    },
                },
            )
            logger.info(f"{response}")
        all_embeddings = self._es_get_all_embeddings()
        nodes = [TextNode(embedding=emb, text="") for emb in all_embeddings]
        self.clustering_by_embedding(nodes)

    def list_all_doc_names(self) -> List[str]:
        """
        List all document names in ES.
        """
        from elasticsearch import Elasticsearch

        all_names = set()
        es_config = self.knowledge_config.get("es_config", {})
        es = Elasticsearch(
            hosts=[es_config["vector_store_args"]["es_url"]],
        )
        response = es.search(
            index=self.knowledge_id,
            body={
                "_source": ["metadata"],
                "query": {
                    "match_all": {},
                },
                "size": 1000,  # Adjust the size according to your needs
            },
            scroll="2m",  # Keep the search context open for 2 minutes
        )
        # Fetch the initial batch of document names
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]

        # Collect the names
        for hit in hits:
            all_names.add(hit["_source"]["metadata"]["file_path"])

        # Continue scrolling until no more documents are found
        while len(hits) > 0:
            response = es.scroll(scroll_id=scroll_id, scroll="2m")
            hits = response["hits"]["hits"]

            for hit in hits:
                all_names.add(hit["_source"]["metadata"]["file_path"])
        return list(all_names)
