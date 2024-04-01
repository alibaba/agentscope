# -*- coding: utf-8 -*-
"""
Unit tests for agent classes and functions
"""

import unittest

from agentscope.agents import AgentBase


class TestAgent(AgentBase):
    """An agent for test usage"""

    def __init__(
        self,
        name: str,
        sys_prompt: str = None,
        **kwargs: dict,
    ) -> None:
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=(
                kwargs["model_config"] if "model_config" in kwargs else None
            ),
            use_memory=(
                kwargs["use_memory"] if "use_memory" in kwargs else None
            ),
            memory_config=(
                kwargs["memory_config"] if "memory_config" in kwargs else None
            ),
            agent_id=(kwargs["agent_id"] if "agent_id" in kwargs else None),
        )


class TestAgentCopy(TestAgent):
    """A copy of testagent"""


class BasicAgentTest(unittest.TestCase):
    """Test cases for basic agents"""

    def test_agent_init(self) -> None:
        """Test the automatic registration mechanism of model wrapper."""
        a1 = TestAgent(
            "a",
            "Hi",
            use_memory=False,  # type: ignore[arg-type]
            attribute_1="hello world",  # type: ignore[arg-type]
        )
        self.assertTupleEqual(
            a1._init_settings["args"],  # pylint: disable=W0212
            (
                "a",
                "Hi",
            ),
        )
        self.assertDictEqual(
            a1._init_settings["kwargs"],  # pylint: disable=W0212
            {"use_memory": False, "attribute_1": "hello world"},
        )
        a2 = TestAgent(
            "b",
            sys_prompt="Hello",
            attribute_2="Bye",  # type: ignore[arg-type]
        )
        self.assertTupleEqual(
            a2._init_settings["args"],  # pylint: disable=W0212
            ("b",),
        )
        self.assertDictEqual(
            a2._init_settings["kwargs"],  # pylint: disable=W0212
            {"sys_prompt": "Hello", "attribute_2": "Bye"},
        )
        self.assertNotEqual(a1.agent_id, a2.agent_id)
        self.assertTrue(a1.agent_id.startswith("TestAgent"))
        self.assertTrue(a2.agent_id.startswith("TestAgent"))
        a3 = TestAgentCopy("c")
        self.assertTrue(a3.agent_id.startswith("TestAgentCopy"))
