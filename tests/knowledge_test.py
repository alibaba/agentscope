# -*- coding: utf-8 -*-
"""
Unit tests for knowledge (RAG module in AgentScope)
"""

import os
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Optional
import shutil
import json

import agentscope
from agentscope.manager import ASManager
from agentscope.models import OpenAIEmbeddingWrapper, ModelResponse
from agentscope.rag import Knowledge, RetrievedChunk


class DummyModel(OpenAIEmbeddingWrapper):
    """
    Dummy model wrapper for testing
    """

    def __init__(self) -> None:
        """dummy init"""

    def __call__(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """dummy call"""
        return ModelResponse(embedding=[[1.0, 2.0]])


class DummyKnowledge(Knowledge):
    """Dummy knowledge class"""

    knowledge_type = "dummy_knowledge_class"

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
    ) -> list[Any]:
        return []

    @classmethod
    def build_knowledge_instance(
        cls,
        knowledge_id: str,
        knowledge_config: Optional[dict] = None,
        **kwargs: Any,
    ) -> Knowledge:
        return cls(
            knowledge_id=knowledge_id,
        )


class KnowledgeTest(unittest.TestCase):
    """
    Test cases for TemporaryMemory
    """

    def setUp(self) -> None:
        """set up test data"""
        agentscope.init(disable_saving=True)

        self.data_dir = "tmp_data_dir"
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.file_name_1 = "tmp_data_dir/file1.txt"
        self.content = "testing file"
        with open(self.file_name_1, "w", encoding="utf-8") as f:
            f.write(self.content)

    def tearDown(self) -> None:
        """Clean up before & after tests."""
        ASManager.get_instance().flush()
        try:
            if os.path.exists(self.data_dir):
                shutil.rmtree(self.data_dir)
            if os.path.exists("./runs"):
                shutil.rmtree("./runs")
            if os.path.exists("./test_knowledge"):
                shutil.rmtree("./test_knowledge")
        except Exception:
            pass

    def test_llamaindexknowledge(self) -> None:
        """test llamaindexknowledge"""
        from agentscope.rag.llama_index_knowledge import LlamaIndexKnowledge

        dummy_model = DummyModel()

        knowledge_config = {
            "knowledge_id": "",
            "data_processing": [],
        }
        loader_config = {
            "load_data": {
                "loader": {
                    "create_object": True,
                    "module": "llama_index.core",
                    "class": "SimpleDirectoryReader",
                    "init_args": {},
                },
            },
        }
        loader_init = {"input_dir": self.data_dir, "required_exts": ".txt"}

        loader_config["load_data"]["loader"]["init_args"] = loader_init
        knowledge_config["data_processing"].append(loader_config)

        knowledge = LlamaIndexKnowledge(
            knowledge_id="test_knowledge",
            emb_model=dummy_model,
            knowledge_config=knowledge_config,
        )
        retrieved = knowledge.retrieve(
            query="testing",
            similarity_top_k=2,
            to_list_strs=True,
        )
        self.assertEqual(
            retrieved,
            [self.content],
        )

    @patch("agentscope.utils.common.requests.get")
    def test_bingknowledge(self, mock_get: MagicMock) -> None:
        """
        Test BingKnowledge function
        """
        from agentscope.rag.search_knowledge import BingKnowledge

        mock_response = Mock()
        bing_title = "Test title from Bing"
        bing_snippet = "Test snippet from Bing"
        bing_return_url = "Test url from Bing"
        mock_dict = {
            "webPages": {
                "value": [
                    {
                        "name": bing_title,
                        "url": bing_return_url,
                        "snippet": bing_snippet,
                    },
                ],
            },
        }
        mock_response.json.return_value = mock_dict
        mock_get.return_value = mock_response

        os.environ["BING_SEARCH_KEY"] = "xxx"
        info = {
            "Title": bing_title,
            "Description": bing_snippet,
            "Reference": bing_return_url,
        }
        content = json.dumps(info, ensure_ascii=False)
        expected_result = [
            RetrievedChunk(
                content=content,
                metadata={"file_path": bing_return_url},
                score=1.0,
            ),
        ]
        search_knowledge = BingKnowledge(
            knowledge_id="test_knowledge",
            knowledge_config={},
            to_load_web=False,
        )
        retrieved = search_knowledge.retrieve("test", similarity_top_k=1)
        self.assertEqual(
            retrieved,
            expected_result,
        )

    def test_knowledge_bank(self) -> None:
        """test knowledge bank"""
        dummy_knowledge_id = "test_dummy_knowledge"
        knowledge_configs = [
            {
                "knowledge_id": dummy_knowledge_id,
                "knowledge_type": "dummy_knowledge_class",
            },
        ]
        from agentscope.rag.knowledge_bank import KnowledgeBank

        knowledge_bank = KnowledgeBank(
            configs=knowledge_configs,
            new_knowledge_types=[DummyKnowledge],
        )
        dummy_knowledge = knowledge_bank.get_knowledge(dummy_knowledge_id)
        self.assertEqual(
            dummy_knowledge.retrieve("test", similarity_top_k=1),
            [],
        )
        # test not allow duplicate register knowledge type
        self.assertRaises(
            ValueError,
            knowledge_bank.register_knowledge_type,
            DummyKnowledge,
            False,
        )


if __name__ == "__main__":
    unittest.main()
