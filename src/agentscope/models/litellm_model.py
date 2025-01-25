# -*- coding: utf-8 -*-
"""Model wrapper based on litellm https://docs.litellm.ai/docs/"""
from abc import ABC
from typing import Union, Any, List, Sequence, Optional, Generator

from loguru import logger

from ._model_utils import _verify_text_content_in_openai_delta_response
from .model import ModelWrapperBase, ModelResponse
from ..message import Msg


class LiteLLMWrapperBase(ModelWrapperBase, ABC):
    """The model wrapper based on LiteLLM API."""

    def __init__(
        self,
        config_name: str,
        model_name: str = None,
        generate_args: dict = None,
        **kwargs: Any,
    ) -> None:
        """
        To use the LiteLLM wrapper, environment variables must be set.
        Different model_name could be using different environment variables.
        For example:
            - for model_name: "gpt-3.5-turbo", you need to set "OPENAI_API_KEY"
            ```
            os.environ["OPENAI_API_KEY"] = "your-api-key"
            ```
            - for model_name: "claude-2", you need to set "ANTHROPIC_API_KEY"
            - for Azure OpenAI, you need to set "AZURE_API_KEY",
            "AZURE_API_BASE", "AZURE_API_VERSION"
        You should refer to the docs in https://docs.litellm.ai/docs/

        Args:
            config_name (`str`):
                The name of the model config.
            model_name (`str`, default `None`):
                The name of the model to use in OpenAI API.
            generate_args (`dict`, default `None`):
                The extra keyword arguments used in litellm api generation,
                e.g. `temperature`, `seed`.
                For generate_args, please refer to
                https://docs.litellm.ai/docs/completion/input
                for more details.

        """

        if model_name is None:
            model_name = config_name
            logger.warning("model_name is not set, use config_name instead.")

        super().__init__(config_name=config_name, model_name=model_name)

        self.generate_args = generate_args or {}

    def format(
        self,
        *args: Union[Msg, Sequence[Msg]],
    ) -> Union[List[dict], str]:
        raise RuntimeError(
            f"Model Wrapper [{type(self).__name__}] doesn't "
            f"need to format the input. Please try to use the "
            f"model wrapper directly.",
        )


class LiteLLMChatWrapper(LiteLLMWrapperBase):
    """The model wrapper based on litellm chat API.

    Note:
        - litellm requires the users to set api key in their environment
        - Different LLMs requires different environment variables

    Example:
        - For OpenAI models, set "OPENAI_API_KEY"
        - For models like "claude-2", set "ANTHROPIC_API_KEY"
        - For Azure OpenAI models, you need to set "AZURE_API_KEY",
        "AZURE_API_BASE" and "AZURE_API_VERSION"
        - Refer to the docs in https://docs.litellm.ai/docs/ .


    Response:
        - From https://docs.litellm.ai/docs/completion/output

        .. code-block:: json

            {
                'choices': [
                    {
                        'finish_reason': str,  # String: 'stop'
                        'index': int,  # Integer: 0
                        'message': {  # Dictionary [str, str]
                            'role': str,  # String: 'assistant'
                            'content': str  # String: "default message"
                        }
                    }
                ],
                'created': str,  # String: None
                'model': str,  # String: None
                'usage': {  # Dictionary [str, int]
                    'prompt_tokens': int,  # Integer
                    'completion_tokens': int,  # Integer
                    'total_tokens': int  # Integer
                }
            }

    """

    model_type: str = "litellm_chat"

    def __init__(
        self,
        config_name: str,
        model_name: str = None,
        stream: bool = False,
        generate_args: dict = None,
        **kwargs: Any,
    ) -> None:
        """
        To use the LiteLLM wrapper, environment variables must be set.
        Different model_name could be using different environment variables.
        For example:
            - for model_name: "gpt-3.5-turbo", you need to set "OPENAI_API_KEY"
            ```
            os.environ["OPENAI_API_KEY"] = "your-api-key"
            ```
            - for model_name: "claude-2", you need to set "ANTHROPIC_API_KEY"
            - for Azure OpenAI, you need to set "AZURE_API_KEY",
            "AZURE_API_BASE", "AZURE_API_VERSION"
        You should refer to the docs in https://docs.litellm.ai/docs/

        Args:
            config_name (`str`):
                The name of the model config.
            model_name (`str`, default `None`):
                The name of the model to use in OpenAI API.
            stream (`bool`, default `False`):
                Whether to enable stream mode.
            generate_args (`dict`, default `None`):
                The extra keyword arguments used in litellm api generation,
                e.g. `temperature`, `seed`.
                For generate_args, please refer to
                https://docs.litellm.ai/docs/completion/input
                for more details.

        """

        super().__init__(
            config_name=config_name,
            model_name=model_name,
            generate_args=generate_args,
            **kwargs,
        )

        self.stream = stream

    def __call__(
        self,
        messages: list,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Args:
            messages (`list`):
                A list of messages to process.
            stream (`Optional[bool]`, default `None`):
                Whether to enable stream mode. If not set, the stream mode
                will be set to the value in the initialization.
            **kwargs (`Any`):
                The keyword arguments to litellm chat completions API,
                e.g. `temperature`, `max_tokens`, `top_p`, etc. Please refer to
                https://docs.litellm.ai/docs/completion/input
                for more detailed arguments.

        Returns:
            `ModelResponse`:
                The response text in text field, and the raw response in
                raw field.
        """

        # step1: prepare keyword arguments
        kwargs = {**self.generate_args, **kwargs}

        # step2: checking messages
        if not isinstance(messages, list):
            raise ValueError(
                "LiteLLM `messages` field expected type `list`, "
                f"got `{type(messages)}` instead.",
            )
        if not all("role" in msg and "content" in msg for msg in messages):
            raise ValueError(
                "Each message in the 'messages' list must contain a 'role' "
                "and 'content' key for LiteLLM API.",
            )

        # Import litellm only when it is used
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "Cannot find litellm in current environment, please "
                "install it by `pip install litellm`.",
            ) from e

        # step3: forward to generate response
        if stream is None:
            stream = self.stream

        kwargs.update(
            {
                "model": self.model_name,
                "messages": messages,
                "stream": stream,
            },
        )

        # Add stream_options to obtain the usage information
        if stream:
            kwargs["stream_options"] = {"include_usage": True}

        response = litellm.completion(**kwargs)

        if stream:

            def generator() -> Generator[str, None, None]:
                text = ""
                last_chunk = {}
                for chunk in response:
                    # In litellm, the content maybe `None` for the last second
                    # chunk
                    chunk = chunk.model_dump()
                    if _verify_text_content_in_openai_delta_response(chunk):
                        text += chunk["choices"][0]["delta"]["content"]
                        yield text
                    last_chunk = chunk

                # Update the last chunk to save locally
                if last_chunk.get("choices", []) in [None, []]:
                    last_chunk["choices"] = [{}]

                last_chunk["choices"][0]["message"] = {
                    "role": "assistant",
                    "content": text,
                }

                self._save_model_invocation_and_update_monitor(
                    kwargs,
                    last_chunk,
                )

            return ModelResponse(
                stream=generator(),
            )

        else:
            response = response.model_dump()
            self._save_model_invocation_and_update_monitor(
                kwargs,
                response,
            )

            # return response
            return ModelResponse(
                text=response["choices"][0]["message"]["content"],
                raw=response,
            )

    def _save_model_invocation_and_update_monitor(
        self,
        kwargs: dict,
        response: dict,
    ) -> None:
        """Save the model invocation and update the monitor accordingly."""

        # step4: record the api invocation if needed
        self._save_model_invocation(
            arguments=kwargs,
            response=response,
        )

        # step5: update monitor accordingly
        usage = response.get("usage", None)
        if usage is not None:
            self.monitor.update_text_and_embedding_tokens(
                model_name=self.model_name,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

    def format(
        self,
        *args: Union[Msg, Sequence[Msg]],
    ) -> List[dict]:
        """A common format strategy for chat models, which will format the
        input messages into a user message.

        Note this strategy maybe not suitable for all scenarios,
        and developers are encouraged to implement their own prompt
        engineering strategies.

        The following is an example:

        .. code-block:: python

            prompt1 = model.format(
                Msg("system", "You're a helpful assistant", role="system"),
                Msg("Bob", "Hi, how can I help you?", role="assistant"),
                Msg("user", "What's the date today?", role="user")
            )

            prompt2 = model.format(
                Msg("Bob", "Hi, how can I help you?", role="assistant"),
                Msg("user", "What's the date today?", role="user")
            )

        The prompt will be as follows:

        .. code-block:: python

            # prompt1
            [
                {
                    "role": "system",
                    "content": "You're a helpful assistant"
                },
                {
                    "role": "user",
                    "content": (
                        "## Conversation History\\n"
                        "Bob: Hi, how can I help you?\\n"
                        "user: What's the date today?"
                    )
                }
            ]

            # prompt2
            [
                {
                    "role": "user",
                    "content": (
                        "## Conversation History\\n"
                        "Bob: Hi, how can I help you?\\n"
                        "user: What's the date today?"
                    )
                }
            ]


        Args:
            args (`Union[Msg, Sequence[Msg]]`):
                The input arguments to be formatted, where each argument
                should be a `Msg` object, or a list of `Msg` objects.
                In distribution, placeholder is also allowed.

        Returns:
            `List[dict]`:
                The formatted messages.
        """

        return ModelWrapperBase.format_for_common_chat_models(*args)


class LiteLLMVisionWrapper(LiteLLMChatWrapper):
    """The model wrapper based on litellm chat API with vision capabilities.

    This class extends the LiteLLMChatWrapper to support multimodal inputs,
    including both text and images. It is designed to work with vision-language
    models that can process and respond to both textual and visual information.

    reference:
    https://docs.litellm.ai/docs/completion/vision#checking-if-a-model-supports-vision

    Note:
        - The model used must support vision capabilities (e.g., GPT-4o).

    Example:
        To use this wrapper with a vision-capable model:
        1. specify "model_type" as "litellm_chat_v".
        2. give the url of the image in message in the following way:
        ```python
        Msg(
            name="Alice",
            content="what is the image about",
            role="user",
            url="https://xxx.jpg",
        )
        ```


    Response:
        The response format is the same as LiteLLMChatWrapper,
        but the model can now process and respond to both
        text and image inputs.
    """

    model_type: str = "litellm_chat_v"

    def __init__(
        self,
        config_name: str,
        model_name: str = None,
        **kwargs: Any,
    ) -> None:
        if model_name is None:
            model_name = config_name
            logger.warning("model_name is not set, use config_name instead.")

        super().__init__(config_name, model_name, **kwargs)

    def format(self, *args: Union[Msg, Sequence[Msg]]) -> List:
        """Format the input messages for vision-language models.

        This method processes a sequence of Msg objects, handling
        both text and image content, and formats them into a
        structure suitable for vision-language models.

        Args:
            *args (Union[Msg, Sequence[Msg]]): A sequence of Msg objects
                                               or lists of Msg objects.

        Returns:
            List: A list of formatted messages ready for the
                  vision-language model.

        Raises:
            TypeError: If the input is not a Msg object or a list
                       of Msg objects.

        Note:
            - For 'system' role messages, only text content is allowed.
            - For other roles, both text and image content can be included.
            - Image content is expected to be provided as a URL in the
              Msg object's 'url' field.
        """
        input_msgs = []
        for item in args:
            if item is None:
                continue
            if isinstance(item, Msg):
                input_msgs.append(item)
            elif isinstance(item, list) and all(
                isinstance(subitem, Msg) for subitem in item
            ):
                input_msgs.extend(item)
            else:
                raise TypeError(
                    "The input should be a Msg object or "
                    f"a list of Msg objects, got {type(item)}.",
                )

        messages = []

        for msg in input_msgs:
            if msg.role == "system":
                # For 'system' role, set 'content' directly to msg.content
                content = msg.content
            else:
                formatted_content = []
                if msg.content:  # Handle text content
                    formatted_content.append(
                        {
                            "type": "text",
                            "text": msg.content,
                        },
                    )

                if msg.url:  # Handle image URL content
                    formatted_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": msg.url,
                            },
                        },
                    )
                content = formatted_content

            messages.append(
                {
                    "role": msg.role,
                    "content": content,
                },
            )

        return messages
