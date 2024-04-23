# -*- coding: utf-8 -*-
""" Unit test for service factory. """
import json
import unittest
from typing import Literal

from agentscope.models import ModelWrapperBase
from agentscope.service import (
    bing_search,
    execute_python_code,
    retrieve_from_list,
    query_mysql,
    summarization,
)
from agentscope.service.service_factory import ServiceFactory


class ServiceFactoryTest(unittest.TestCase):
    """
    Unit test for service factory.
    """

    def setUp(self) -> None:
        """Init for ExampleTest."""
        self.json_schema_bing_search1 = {
            "type": "function",
            "function": {
                "name": "bing_search",
                "description": (
                    "Search question in Bing Search API and "
                    "return the searching results"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "num_results": {
                            "type": "number",
                            "description": (
                                "The number of search " "results to return."
                            ),
                            "default": 10,
                        },
                        "question": {
                            "type": "string",
                            "description": "The search query string.",
                        },
                    },
                    "required": [
                        "question",
                    ],
                },
            },
        }

        self.json_schema_bing_search2 = {
            "type": "function",
            "function": {
                "name": "bing_search",
                "description": (
                    "Search question in Bing Search API and "
                    "return the searching results"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The search query string.",
                        },
                    },
                    "required": ["question"],
                },
            },
        }

        self.json_schema_func = {
            "type": "function",
            "function": {
                "name": "func",
                "description": None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "c": {"default": "test"},
                        "d": {
                            "type": "typing.Literal",
                            "enum": [1, "abc", "d"],
                            "default": 1,
                        },
                        "b": {},
                        "a": {"type": "string"},
                    },
                    "required": ["a", "b"],
                },
            },
        }

        self.json_schema_execute_python_code = {
            "type": "function",
            "function": {
                "name": "execute_python_code",
                "description": "Execute a piece of python code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": (
                                "The Python code to be " "executed."
                            ),
                        },
                    },
                    "required": ["code"],
                },
            },
        }

        self.json_schema_retrieve_from_list = {
            "type": "function",
            "function": {
                "name": "retrieve_from_list",
                "description": "Retrieve data in a list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "description": "A message to be retrieved.",
                        },
                    },
                    "required": [
                        "query",
                    ],
                },
            },
        }

        self.json_schema_query_mysql = {
            "type": "function",
            "function": {
                "name": "query_mysql",
                "description": "Execute query within MySQL database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute.",
                        },
                    },
                    "required": [
                        "query",
                    ],
                },
            },
        }

        self.json_schema_summarization = {
            "type": "function",
            "function": {
                "name": "summarization",
                "description": "Summarize the input text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": (
                                "Text to be summarized by " "the model."
                            ),
                        },
                    },
                    "required": [
                        "text",
                    ],
                },
            },
        }

    def test_bing_search(self) -> None:
        """Test bing_search."""
        # api_key is specified by developer, while question and num_results
        # are specified by model
        _, doc_dict = ServiceFactory.get(bing_search, api_key="xxx")
        print(json.dumps(doc_dict, indent=4))
        self.assertDictEqual(
            doc_dict,
            self.json_schema_bing_search1,
        )

        # Set num_results by developer rather than model
        _, doc_dict = ServiceFactory.get(
            bing_search,
            num_results=3,
            api_key="xxx",
        )

        self.assertDictEqual(
            doc_dict,
            self.json_schema_bing_search2,
        )

    def test_enum(self) -> None:
        """Test enum in service factory."""

        def func(  # type: ignore
            a: str,
            b,
            c="test",
            d: Literal[1, "abc", "d"] = 1,
        ) -> None:
            print(a, b, c, d)

        _, doc_dict = ServiceFactory.get(func)

        self.assertDictEqual(
            doc_dict,
            self.json_schema_func,
        )

    def test_exec_python_code(self) -> None:
        """Test execute_python_code in service factory."""
        _, doc_dict = ServiceFactory.get(
            execute_python_code,
            timeout=300,
            use_docker=True,
            maximum_memory_bytes=None,
        )

        self.assertDictEqual(
            doc_dict,
            self.json_schema_execute_python_code,
        )

    def test_retrieval(self) -> None:
        """Test retrieval in service factory."""
        _, doc_dict = ServiceFactory.get(
            retrieve_from_list,
            knowledge=[1, 2, 3],
            score_func=lambda x, y: 1.0,
            top_k=10,
            embedding_model=10,
            preserve_order=True,
        )

        self.assertDictEqual(
            doc_dict,
            self.json_schema_retrieve_from_list,
        )

    def test_sql_query(self) -> None:
        """Test sql_query in service factory."""
        _, doc_dict = ServiceFactory.get(
            query_mysql,
            database="test",
            host="localhost",
            user="root",
            password="xxx",
            port=3306,
            allow_change_data=False,
            maxcount_results=None,
        )

        self.assertDictEqual(
            doc_dict,
            self.json_schema_query_mysql,
        )

    def test_summary(self) -> None:
        """Test summarization in service factory."""
        _, doc_dict = ServiceFactory.get(
            summarization,
            model=ModelWrapperBase("abc"),
            system_prompt="",
            summarization_prompt="",
            max_return_token=-1,
            token_limit_prompt="",
        )

        print(json.dumps(doc_dict, indent=4))

        self.assertDictEqual(
            doc_dict,
            self.json_schema_summarization,
        )

    def test_object_service_factory(self) -> None:
        """Test the object of ServiceFactory."""
        service_factory = ServiceFactory()

        service_factory.add(bing_search, api_key="xxx", num_results=3)
        service_factory.add(
            execute_python_code,
            timeout=300,
            use_docker=True,
            maximum_memory_bytes=None,
        )

        self.assertEqual(
            service_factory.tools_instruction,
            """## Tool Functions:
The following tool functions are available in the format of
```
{index}. {function name}: {function description}
{argument1 name} ({argument type}): {argument description}
{argument2 name} ({argument type}): {argument description}
...
```

1. bing_search: Search question in Bing Search API and return the searching results
	question (string): The search query string.
2. execute_python_code: Execute a piece of python code.
	code (string): The Python code to be executed.
""",  # noqa
        )
        self.assertDictEqual(
            service_factory.json_schemas,
            {
                "bing_search": self.json_schema_bing_search2,
                "execute_python_code": self.json_schema_execute_python_code,
            },
        )


if __name__ == "__main__":
    unittest.main()
