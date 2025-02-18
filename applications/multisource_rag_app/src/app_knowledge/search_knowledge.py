# -*- coding: utf-8 -*-
# pylint: disable=E0611,W0212
"""
Enhance search knowledge with embedding-based routing
"""
from typing import Union, Sequence, Optional, Any, List
import json
import os
import copy
from elasticsearch import Elasticsearch

from utils.logging import logger

from agentscope.rag import Knowledge, RetrievedChunk
from agentscope.service import bing_search, load_web
from agentscope.tokens import count_openai_tokens
from agentscope.constants import DEFAULT_CHUNK_SIZE


class BingKnowledge(Knowledge):
    """Enhanced BingKnowledge that can be used to embedding-based routing"""

    knowledge_type = "enhanced_bing_knowledge"

    def __init__(
        self,
        knowledge_id: str,
        knowledge_config: Optional[dict] = None,
        to_load_web: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        This class is a knowledge wrapper using Bing Search to
        retrieve information online.

        Args:
            knowledge_id (`str`):
                The id of the RAG knowledge unit.
            knowledge_config (`dict`):
                The configuration for using online search.
                The knowledge config for BingKnowledge can have the following
                parameter keys:
                "query_prefix":
                    some prefix for searching, for example, enforce search in
                    some domain
                Other parameters in https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters # noqa: E501 #pylint: disable=line-too-long
                can also be added to the knowledge config.
                For example,
                {
                    "query_prefix": "site:community.modelscope.cn ",
                    "freshness": "Month",
                    # Return webpages that Bing discovered within the
                    # last 30 days.
                }
            load_web (`bool`):
                Whether to load the web page content. Currently only supports
                loading content in "p" tags
        """
        super().__init__(
            knowledge_id=knowledge_id,
            knowledge_config=knowledge_config,
            **kwargs,
        )
        self.to_load_web = to_load_web

    def _init_rag(
        self,
        **kwargs: Any,
    ) -> Any:
        pass

    def retrieve(
        self,
        query: Any,
        similarity_top_k: int = None,
        to_list_strs: bool = False,
        **kwargs: Any,
    ) -> List[Union[RetrievedChunk, str]]:
        """
        Retrieve content from bing search and format them in list.

        Args:
            query (`Any`):
                User query to search on
            similarity_top_k (`int`):
                Number of search results to return
            to_list_strs (`bool`):
                Whether to return list of strings or
                return list of RetrievedChunk

        Returns:
            `List[Union[RetrievedChunk, str]]`: List of retrieved content
        """
        search_config = copy.deepcopy(self.knowledge_config)
        query_prefix = search_config.pop("query_prefix", "")
        query = query_prefix + " " + query
        logger.info(f"bing query: {query}")
        key = os.getenv("BING_SEARCH_KEY", "")
        if len(key) == 0:
            raise ValueError(
                "Bing search key is empty."
                "Need to set bing key as environment variable:"
                "`export BING_SEARCH_KEY=xxxx` ",
            )
        bing_result = bing_search(
            query,
            api_key=key,
            num_results=similarity_top_k,
            **search_config,
        ).content
        chunk_list = []
        logger.info(f"bing result: {bing_result}")
        if not isinstance(bing_result, list):
            return chunk_list

        for info_piece in bing_result:
            url = info_piece.get("link", "")
            bing_title = info_piece.get("title", "")
            info = {
                "Title": bing_title,
                "Description": info_piece.get("snippet", ""),
            }

            if self.to_load_web:
                try:
                    extracted = load_web(
                        url,
                        html_selected_tags=["p"],
                    ).content["html_to_text"]
                except (NotImplementedError, TypeError, KeyError) as e:
                    logger.warning(f"Load web page fail: {e}")
                    extracted = ""
                messages = [
                    {"assistant": extracted},
                ]
                try:
                    token = count_openai_tokens(
                        model_name="gpt-4-0613",
                        messages=messages,
                    )
                except NotImplementedError as e:
                    logger.warning(e)
                    continue
                if token > DEFAULT_CHUNK_SIZE - 100:
                    extracted = extracted[: DEFAULT_CHUNK_SIZE - 100]
                info["additional_info"] = extracted

            info["Reference"] = url
            content = json.dumps(info, ensure_ascii=False)
            chunk_list.append(
                RetrievedChunk(
                    content=content,
                    metadata={"file_path": url},
                    score=1.0,
                ),
            )

        if to_list_strs:
            results = []
            for chunk in chunk_list:
                results.append(str(chunk.content))
            return results

        return chunk_list

    @classmethod
    def default_config(cls, **kwargs: Any) -> dict:
        """
        Return a default config for a knowledge class.
        """
        return {}

    @classmethod
    def build_knowledge_instance(
        cls,
        knowledge_id: str,
        knowledge_config: Optional[dict] = None,
        to_load_web: bool = False,
        **kwargs: Any,
    ) -> "Knowledge":
        """
        A constructor to build a knowledge base instance.

        Args:
            knowledge_id (`str`):
                The id of the knowledge instance.
            knowledge_config (`dict`):
                The configuration to the knowledge instance.
            to_load_web (`bool`):
                Whether to load the web page content.

        Returns:
            `Knowledge`: A BingKnowledge instance.
        """
        return cls(
            knowledge_id=knowledge_id,
            knowledge_config=knowledge_config,
            to_load_web=to_load_web,
        )

    def cluster_similarities(
        self,
        query_or_embedding: Union[str, Sequence[float]],
    ) -> Sequence:
        """
        Return similarities between query and embedding
        When there is possible that there is some prepared cluster means for
        this knowledge, use the prepared means to computer;
        otherwise return empty list.
        """

        def _cluster_means_idx_name() -> str:
            """return hard-coded name of the index in Elasticsearch"""
            return self.knowledge_id + "_cluster_means"

        similarity = []
        if "es_config" in self.knowledge_config:
            logger.info("Using ES for bing search routing")

            if isinstance(query_or_embedding, str):
                query_embedding = (
                    self.emb_model._get_query_embedding(  # noqa: W0212
                        query_or_embedding,
                    )
                )
            else:
                query_embedding = query_or_embedding

            cluster_idx_name = _cluster_means_idx_name()
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
                        # Number of nearest neighbors to return
                        "k": 5,
                        # Number of candidate vectors to consider
                        "num_candidates": 1000,
                    },
                },
            )
            for hit in response["hits"]["hits"]:
                similarity.append(hit["_score"])

        logger.info(
            "[DONE] clustering in Bing search knowledge, "
            f"len of sim: {len(similarity)}",
        )
        return similarity
