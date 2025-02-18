# -*- coding: utf-8 -*-
"""
Rerank function for mixed multi-source retrieved knowledge
"""
import os
from typing import List, Optional
from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import (
    MetadataMode,
    NodeWithScore,
    QueryBundle,
    TextNode,
)

from agentscope.rag import RetrievedChunk

dispatcher = get_dispatcher()
try:
    import dashscope
except ImportError as e:
    raise ImportError(f"{e}:DashScope requires `pip install dashscope`") from e


class DashScopeRerank(BaseNodePostprocessor):
    """
    Dashscope rerank model processor
    """

    model: str = Field(description="Dashscope rerank model name.")
    top_n: int = Field(description="Top N nodes to return.")

    def __init__(
        self,
        top_n: int = 3,
        model: str = "gte-rerank",
        return_documents: bool = False,
    ):
        """
        Initialize rerank model with parameters
        Args:
            top_n: Top N nodes to return
            model: Dashscope rerank model name, default is "gte-rerank"
            return_documents: Whether return documentation after reranking
        """
        super().__init__(
            top_n=top_n,
            model=model,
            return_documents=return_documents,
        )
        assert (
            os.getenv("DASHSCOPE_API_KEY") is not None
        ), "Dashscope API key required"

    @classmethod
    def class_name(cls) -> str:
        """
        Overwrite the llamaindex function
        """
        return "DashScopeRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        The post-processing nodes function
        Args:
            nodes: List of nodes to rerank
            query_bundle: Query (messages from user) bundle
        """
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []
        texts = [
            node.node.get_content(metadata_mode=MetadataMode.NONE)
            for node in nodes
        ]
        results = dashscope.TextReRank.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=self.model,
            top_n=self.top_n,
            query=query_bundle.query_str,
            documents=texts,
        )
        new_nodes = []
        for result in results.output.results:
            new_node_with_score = NodeWithScore(
                node=nodes[result.index].node,
                score=result.relevance_score,
            )
            new_nodes.append(new_node_with_score)
        return new_nodes


def ds_rerank(
    query: str,
    chunks: List[RetrievedChunk],
    top_k: int,
) -> List[RetrievedChunk]:
    """
    A wrapper function for dashscope rerank function
    Args:
        query: User query
        chunks: List of nodes to rerank
        top_k: Top N nodes to return
    """
    dashscope_rerank = DashScopeRerank(top_n=top_k)
    nodes = []
    for chunk in chunks:
        node = NodeWithScore(node=TextNode(), score=chunk.score)
        node.node.set_content(chunk.content)
        node.node.embedding = chunk.embedding
        node.node.metadata = chunk.metadata
        nodes.append(node)
    try:
        ranked_nodes = dashscope_rerank.postprocess_nodes(
            nodes,
            query_str=query,
        )
    except AttributeError as exc:
        print(f"rerank fail: {exc}")
        return chunks

    # convert back to RetrievedChunk
    ranked_chunks = []
    for node in ranked_nodes:
        ranked_chunks.append(
            RetrievedChunk(
                score=node.score,
                content=node.node.get_content(),
                metadata=node.metadata,
                embedding=node.embedding,
                hash=node.node.hash,
            ),
        )
    return ranked_chunks
