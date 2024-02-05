# -*- coding: utf-8 -*-
""" Unit test for logger chat"""
import shutil
import time
import unittest

from loguru import logger

from agentscope.utils import setup_logger


class LoggerTest(unittest.TestCase):
    """
    Unit test for logger.
    """

    def test_logger_chat(self) -> None:
        """Logger chat."""

        setup_logger("./runs/", level="INFO")

        # str with "\n"
        logger.chat("Test\nChat\n\nMessage\n\n")

        # dict with "\n"
        logger.chat(
            {
                "name": "Alice",
                "content": "Hi!\n",
                "url": "https://xxx.png",
            },
        )

        # dict without content
        logger.chat({"name": "Alice", "url": "https://xxx.png"})

        # dict
        logger.chat({"abc": 1})

        # To avoid that logging is not finished before the file is read
        time.sleep(3)

        with open("./runs/logging.chat", "r", encoding="utf-8") as file:
            lines = file.readlines()

        ground_truth = [
            '"Test\\nChat\\n\\nMessage\\n\\n"\n',
            '{"name": "Alice", "content": "Hi!\\n", "url": "https://xxx.png'
            '"}\n',
            '{"name": "Alice", "url": "https://xxx.png"}\n',
            '{"abc": 1}\n',
        ]

        self.assertListEqual(lines, ground_truth)

    def tearDown(self) -> None:
        """Tear down for LoggerTest."""
        shutil.rmtree("./runs/")


if __name__ == "__main__":
    unittest.main()
