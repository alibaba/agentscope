# -*- coding: utf-8 -*-
import time
import json
from typing import Optional, Union, Any, Callable
from loguru import logger

from agentscope.agents import AgentBase
from agentscope.message import Msg


class RuledUser(AgentBase):
    """User agent under rules"""

    def __init__(
        self,
        name: str = "User",
        model: Optional[Union[Callable[..., Any], str]] = None,
        sys_prompt: Optional[str] = None,
    ) -> None:
        """Initialize a RuledUser object."""
        super().__init__(name=name, model=model, sys_prompt=sys_prompt)
        self.retry_time = 10

    def reply(
        self,
        x: dict = None,
        required_keys: Optional[Union[list[str], str]] = None,
    ) -> dict:
        """
        Processes the input provided by the user and stores it in memory,
        potentially formatting it with additional provided details.
        """
        if x is not None:
            self.memory.add(x)

        # TODO: To avoid order confusion, because `input` print much quicker
        #  than logger.chat
        time.sleep(0.5)
        while True:
            try:
                content = input(f"{self.name}: ")
                if not hasattr(self, "model"):
                    break

                ruler_res = self.is_content_valid(content)
                if ruler_res.get("allowed") == "true":
                    break
                else:
                    logger.warning(
                        f"Input is not allowed:"
                        f" {ruler_res.get('reason', 'Unknown reason')}. "
                        f"Please retry.",
                    )
            except Exception as e:
                logger.warning(f"Input invalid: {e}. Please try again.")

        kwargs = {}
        if required_keys is not None:
            if isinstance(required_keys, str):
                required_keys = [required_keys]

            for key in required_keys:
                kwargs[key] = input(f"{key}: ")

        # Add additional keys
        msg = Msg(
            self.name,
            role="user",
            content=content,
            **kwargs,  # type: ignore[arg-type]
        )

        # Add to memory
        self.memory.add(msg)

        return msg

    def is_content_valid(self, content):
        prompt = self.sys_prompt.format_map({"content": content})
        message = Msg(name="user", content=prompt, role="user")
        ruler_res = self.model(
            messages=[message],
            parse_func=json.loads,
            max_retries=self.retry_time,
        )
        return ruler_res
