# -*- coding: utf-8 -*-
"""AgentScope exception classes."""

# - Model Response Parsing Exceptions


class ResponseParsingError(Exception):
    """The exception class for response parsing error with uncertain
    reasons."""

    raw_response: str
    """Record the raw response."""

    def __init__(self, message: str, raw_response: str = None) -> None:
        """Initialize the exception with the message."""
        self.message = message
        self.raw_response = raw_response

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"


class JsonParsingError(ResponseParsingError):
    """The exception class for JSON parsing error."""


class TagNotFoundError(ResponseParsingError):
    """The exception class for missing tagged content in model response."""

    missing_begin_tag: bool
    """If the response misses the begin tag."""

    missing_end_tag: bool
    """If the response misses the end tag."""

    def __init__(
        self,
        message: str,
        raw_response: str = None,
        missing_begin_tag: bool = True,
        missing_end_tag: bool = True,
    ):
        """Initialize the exception with the message.

        Args:
            raw_response (`str`):
                Record the raw response from the model.
            missing_begin_tag (`bool`, defaults to `True`):
                If the response misses the beginning tag, default to `True`.
            missing_end_tag (`bool`, defaults to `True`):
                If the response misses the end tag, default to `True`.
        """
        super().__init__(message, raw_response)

        self.missing_begin_tag = missing_begin_tag
        self.missing_end_tag = missing_end_tag


# - Function Calling Exceptions


class FunctionCallError(Exception):
    """The base class for exception raising during calling functions."""

    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"


class FunctionCallFormatError(FunctionCallError):
    """The exception class for function calling format error."""


class FunctionNotFoundError(FunctionCallError):
    """The exception class for function not found error."""


class ArgumentNotFoundError(FunctionCallError):
    """The exception class for missing argument error."""


class ArgumentTypeError(FunctionCallError):
    """The exception class for argument type error."""
