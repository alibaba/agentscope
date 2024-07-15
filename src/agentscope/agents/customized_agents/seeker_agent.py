# -*- coding: utf-8 -*-
"""A general dialog agent."""
from typing import Optional, Union, Sequence

from ...message import Msg
from ..dialog_agent import DialogAgent


class SeekerAgent(DialogAgent):
    """A seeker agent who can seek jobs."""

    def __init__(
        self,
        name: str = "seeker",
        cv: str = "cv",
        sys_prompt: str = "I am a seeker.",
        model_config_name: str = "gpt-3.5-turbo",
    ) -> None:
        """Initialize the dialog agent.

        Arguments:
            name (`str`):
                The name of the agent.
            cv (`str`):
                The cv of the agent.
            sys_prompt (`str`):
                The system prompt of the agent, which can be passed by args
                or hard-coded in the agent.
            model_config_name (`str`):
                The name of the model config, which is used to load model from
                configuration.
        """
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )
        self.cv = cv

    def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        """Reply function of the agent. Processes the input data,
        generates a prompt using the current dialogue memory and system
        prompt, and invokes the language model to produce a response. The
        response is then formatted and added to the dialogue memory.

        Args:
            x (`Optional[Union[Msg, Sequence[Msg]]]`, defaults to `None`):
                The input message(s) to the agent, which also can be omitted if
                the agent doesn't need any input.

        Returns:
            `Msg`: The output message generated by the agent.
        """
        # record the input if needed
        if self.memory:
            self.memory.add(x)

        # prepare prompt
        prompt = self.model.format(
            Msg("system", self.sys_prompt, role="system"),
            Msg(self.name, self.cv, role="assistant"),
            self.memory
            and self.memory.get_memory()
            or x,  # type: ignore[arg-type]
        )

        # call llm and generate response
        response = self.model(prompt).text
        msg = Msg(self.name, response, role="assistant")

        # Print/speak the message in this agent's voice
        self.speak(msg)

        # Record the message in memory
        if self.memory:
            self.memory.add(msg)

        return msg
