# -*- coding: utf-8 -*-
"""The parser for JSON object in the model response."""
import json
from copy import deepcopy
from typing import Optional, Any, List

from loguru import logger

from agentscope.exception import (
    TagNotFoundError,
    JsonParsingError,
    JsonTypeError,
    RequiredFieldNotFoundError,
)
from agentscope.models import ModelResponse
from agentscope.parsers import ParserBase
from agentscope.utils.tools import _join_str_with_comma_and


class MarkdownJsonObjectParser(ParserBase):
    """A parser to parse the response text to a json object."""

    name: str = "json block"
    """The name of the parser."""

    tag_begin: str = "```json"
    """Opening tag for a code block."""

    content_hint: str = "{your_json_object}"
    """The hint of the content."""

    tag_end: str = "```"
    """Closing end for a code block."""

    _format_instruction = (
        "You should respond a json object in a json fenced code block as "
        "follows:\n```json\n{content_hint}\n```"
    )
    """The instruction for the format of the json object."""

    def __init__(self, content_hint: Optional[Any] = None) -> None:
        """Initialize the parser with the content hint.

        Args:
            content_hint (`Optional[Any]`, defaults to `None`):
                The hint used to remind LLM what should be fill between the
                tags. If it is a string, it will be used as the content hint
                directly. If it is a dict, it will be converted to a json
                string and used as the content hint.
        """
        if content_hint is not None:
            if isinstance(content_hint, str):
                self.content_hint = content_hint
            else:
                self.content_hint = json.dumps(
                    content_hint,
                    ensure_ascii=False,
                )

    def parse(self, response: ModelResponse) -> ModelResponse:
        """Parse the response text to a json object, and fill it in the parsed
        field in the response object."""

        # extract the content and try to fix the missing tags by hand
        try:
            extract_text = self._extract_first_content_by_tag(
                response,
                self.tag_begin,
                self.tag_end,
            )
        except TagNotFoundError as e:
            # Try to fix the missing tag error by adding the tag
            try:
                response_copy = deepcopy(response)

                # Fix the missing tags
                if e.missing_begin_tag:
                    response_copy.text = (
                        self.tag_begin + "\n" + response_copy.text
                    )
                if e.missing_end_tag:
                    response_copy.text = response_copy.text + self.tag_end

                # Try again to extract the content
                extract_text = self._extract_first_content_by_tag(
                    response_copy,
                    self.tag_begin,
                    self.tag_end,
                )

                # replace the response with the fixed one
                response.text = response_copy.text

                logger.debug("Fix the missing tags by adding them manually.")

            except TagNotFoundError:
                # Raise the original error if the missing tags cannot be fixed
                raise e from None

        # Parse the content into JSON object
        try:
            parsed_json = json.loads(extract_text)
            response.parsed = parsed_json
            return response
        except json.decoder.JSONDecodeError as e:
            raw_response = f"{self.tag_begin}{extract_text}{self.tag_end}"
            raise JsonParsingError(
                f"The content between {self.tag_begin} and {self.tag_end} "
                f"MUST be a JSON object."
                f'When parsing "{raw_response}", an error occurred: {e}',
                raw_response=raw_response,
            ) from None

    @property
    def format_instruction(self) -> str:
        """Get the format instruction for the json object, if the
        format_example is provided, it will be used as the example.
        """
        return self._format_instruction.format(
            content_hint=self.content_hint,
        )


class MarkdownJsonDictParser(MarkdownJsonObjectParser):
    """A class used to parse a JSON dictionary object in a markdown fenced
    code"""

    name: str = "json block"
    """The name of the parser."""

    tag_begin: str = "```json"
    """Opening tag for a code block."""

    content_hint: str = "{your_json_dictionary}"
    """The hint of the content."""

    tag_end: str = "```"
    """Closing end for a code block."""

    _format_instruction = (
        "You should respond a json object in a json fenced code block as "
        "follows:\n```json\n{content_hint}\n```"
    )
    """The instruction for the format of the json object."""

    required_keys: List[str]
    """A list of required keys in the JSON dictionary object. If the response
    misses any of the required keys, it will raise a
    RequiredFieldNotFoundError."""

    def __init__(
        self,
        content_hint: Optional[Any] = None,
        required_keys: List[str] = None,
    ) -> None:
        """Initialize the parser with the content hint.

        Args:
            content_hint (`Optional[Any]`, defaults to `None`):
                The hint used to remind LLM what should be fill between the
                tags. If it is a string, it will be used as the content hint
                directly. If it is a dict, it will be converted to a json
                string and used as the content hint.
            required_keys (`List[str]`, defaults to `[]`):
                A list of required keys in the JSON dictionary object. If the
                response misses any of the required keys, it will raise a
                RequiredFieldNotFoundError.
        """
        super().__init__(content_hint)

        self.required_keys = required_keys or []

    def parse(self, response: ModelResponse) -> ModelResponse:
        """Parse the text field of the response to a JSON dictionary object,
        store it in the parsed field of the response object, and check if the
        required keys exists.
        """
        # Parse the JSON object
        response = super().parse(response)

        if not isinstance(response.parsed, dict):
            # If not a dictionary, raise an error
            raise JsonTypeError(
                "A JSON dictionary object is wanted, "
                f"but got {type(response.parsed)} instead.",
                response.text,
            )

        # Check if the required keys exist
        keys_missing = []
        for key in self.required_keys:
            if key not in response.parsed:
                keys_missing.append(key)

        if len(keys_missing) != 0:
            raise RequiredFieldNotFoundError(
                f"Missing required "
                f"field{'' if len(keys_missing)==1 else 's'} "
                f"{_join_str_with_comma_and(keys_missing)} in the JSON "
                f"dictionary object.",
                response.text,
            )

        return response
