# -*- coding: utf-8 -*-
"""AgentScope exception classes."""


# TODO: move other exception classes here


class ResponseParsingError(Exception):
    """The exception class for response parsing error with uncertain
    reasons."""

    raw_response: str = None
    """Record the raw response."""

    def __init__(self, message: str, raw_response: str) -> None:
        """Initialize the exception with the message."""
        self.message = message
        self.raw_response = raw_response

    def __str__(self) -> str:
        return self.message


class JsonParsingError(ResponseParsingError):
    """The exception class for JSON parsing error."""


class MissingTagError(ResponseParsingError):
    """The exception class for missing tagged content in model response."""
