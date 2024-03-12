# -*- coding: utf-8 -*-
"""Model wrapper for Ollama models."""
from typing import Sequence, Any

from loguru import logger

from agentscope.models import ModelWrapperBase, ModelResponse
from agentscope.utils import QuotaExceededError, MonitorFactory
from agentscope.utils.monitor import get_full_name

try:
    import ollama
except ImportError:
    ollama = None


class OllamaWrapperBase(ModelWrapperBase):
    """The base class for Ollama model wrappers.

    To use Ollama API, please
    1. First install ollama server from https://ollama.com/download and
    start the server
    2. Pull the model by `ollama pull {model_name}` in terminal
    After that, you can use the ollama API.
    """

    model: str
    """The model name used in ollama API."""

    options: dict
    """A dict contains the options for ollama generation API,
    e.g. {"temperature": 0, "seed": 123}"""

    keep_alive: str
    """Controls how long the model will stay loaded into memory following
    the request."""

    def __init__(
        self,
        config_name: str,
        model: str,
        options: dict = None,
        keep_alive: str = "5m",
    ) -> None:
        """Initialize the model wrapper for Ollama API.

        Args:
            model (`str`):
                The model name used in ollama API.
            options (`dict`, default `None`):
                The extra keyword arguments used in Ollama api generation,
                e.g. `{"temperature": 0., "seed": 123}`.
            keep_alive (`str`, default `5m`):
                Controls how long the model will stay loaded into memory
                following the request.
        """

        super().__init__(config_name=config_name)

        self.model = model
        self.options = options
        self.keep_alive = keep_alive

        self.monitor = None

        self._register_default_metrics()

    def _register_default_metrics(self) -> None:
        """Register metrics to the monitor."""
        raise NotImplementedError(
            "The _register_default_metrics function is not Implemented.",
        )

    # TODO: move into ModelWrapperBase
    def _metric(self, metric_name: str) -> str:
        """Add the class name and model name as prefix to the metric name.

        Args:
            metric_name (`str`):
                The metric name.

        Returns:
            `str`: Metric name of this wrapper.
        """
        return get_full_name(name=metric_name, prefix=self.model)


class OllamaChatWrapper(OllamaWrapperBase):
    """The model wrapper for Ollama chat API."""

    model_type: str = "ollama_chat"

    def __call__(
        self,
        messages: Sequence[dict],
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate response from the given messages.

        Args:
            messages (`Sequence[dict]`):
                A list of messages, each message is a dict contains the `role`
                and `content` of the message.

        Returns:
            `ModelResponse`:
                The response text in `text` field, and the raw response in
                `raw` field.
        """
        # step1: prepare parameters accordingly
        if "options" in kwargs:
            # merge the options
            options = {**self.options, **kwargs["options"]}
        else:
            options = self.options

        keep_alive = kwargs.get("keep_alive", self.keep_alive)

        # step2: forward to generate response
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options=options,
            keep_alive=keep_alive,
            **kwargs,
        )

        # step2: record the api invocation if needed
        self._save_model_invocation(
            arguments={
                "model": self.model,
                "messages": messages,
                "options": options,
                "keep_alive": keep_alive,
                **kwargs,
            },
            json_response=response,
        )

        # step3: monitor the response
        try:
            prompt_tokens = response["prompt_eval_count"]
            completion_tokens = response["eval_count"]
            self.monitor.update(
                {
                    "call_counter": 1,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )
        except (QuotaExceededError, KeyError) as e:
            logger.error(e.message)

        # step4: return response
        return ModelResponse(
            text=response["message"]["content"],
            raw=response,
        )

    def _register_default_metrics(self) -> None:
        """Register metrics to the monitor."""
        self.monitor = MonitorFactory.get_monitor()
        self.monitor.register(
            self._metric("call_counter"),
            metric_unit="times",
        )
        self.monitor.register(
            self._metric("prompt_tokens"),
            metric_unit="tokens",
        )
        self.monitor.register(
            self._metric("completion_tokens"),
            metric_unit="token",
        )
        self.monitor.register(
            self._metric("total_tokens"),
            metric_unit="token",
        )


class OllamaEmbeddingWrapper(OllamaWrapperBase):
    """The model wrapper for Ollama embedding API."""

    model_type: str = "ollama_embedding"

    def __call__(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate embedding from the given prompt.

        Args:
            prompt (`str`):
                The prompt to generate response.
            options (`dict`, default `None`):
                The extra keyword arguments used in Ollama api generation,
                e.g. `{"temperature": 0., "seed": 123}`.

        Returns:
            `ModelResponse`:
                The response embedding in `embedding` field, and the raw
                response in `raw` field.
        """
        # step1: prepare parameters accordingly
        if "options" in kwargs:
            # merge the options
            options = {**self.options, **kwargs["options"]}
        else:
            options = self.options

        keep_alive = kwargs.get("keep_alive", self.keep_alive)

        # step2: forward to generate response
        response = ollama.embeddings(
            model=self.model,
            prompt=prompt,
            options=options,
            keep_alive=keep_alive,
            **kwargs,
        )

        # step3: record the api invocation if needed
        self._save_model_invocation(
            arguments={
                "model": self.model,
                "prompt": prompt,
                "options": options,
                "keep_alive": keep_alive,
                **kwargs,
            },
            json_response=response,
        )

        # step4: monitor the response
        try:
            self.monitor.update(
                {"call_counter": 1},
                prefix=self.model,
            )
        except (QuotaExceededError, KeyError) as e:
            logger.error(e.message)

        # step5: return response
        return ModelResponse(
            embedding=response["embedding"],
            raw=response,
        )

    def _register_default_metrics(self) -> None:
        """Register metrics to the monitor."""
        self.monitor = MonitorFactory.get_monitor()
        self.monitor.register(
            self._metric("call_counter"),
            metric_unit="times",
        )


class OllamaGenerationWrapper(OllamaWrapperBase):
    """The model wrapper for Ollama generation API."""

    model_type: str = "ollama_generation"

    def __call__(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Generate response from the given prompt.

        Args:
            prompt (`str`):
                The prompt to generate response.

        Returns:
            `ModelResponse`:
                The response text in `text` field, and the raw response in
                `raw` field.

        """
        # step1: prepare parameters accordingly
        if "options" in kwargs:
            # merge the options
            options = {**self.options, **kwargs["options"]}
        else:
            options = self.options

        keep_alive = kwargs.get("keep_alive", self.keep_alive)

        # step2: forward to generate response
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options=options,
            keep_alive=keep_alive,
        )

        # step3: record the api invocation if needed
        self._save_model_invocation(
            arguments={
                "model": self.model,
                "prompt": prompt,
                "options": options,
                "keep_alive": keep_alive,
                **kwargs,
            },
            json_response=response,
        )

        # step4: monitor the response
        try:
            prompt_tokens = response["prompt_eval_count"]
            completion_tokens = response["eval_count"]
            self.monitor.update(
                {
                    "call_counter": 1,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )
        except (QuotaExceededError, KeyError) as e:
            logger.error(e.message)

        # step5: return response
        return ModelResponse(
            text=response["response"],
            raw=response,
        )

    def _register_default_metrics(self) -> None:
        """Register metrics to the monitor."""
        self.monitor = MonitorFactory.get_monitor()
        self.monitor.register(
            self._metric("call_counter"),
            metric_unit="times",
        )
        self.monitor.register(
            self._metric("prompt_tokens"),
            metric_unit="tokens",
        )
        self.monitor.register(
            self._metric("completion_tokens"),
            metric_unit="token",
        )
        self.monitor.register(
            self._metric("total_tokens"),
            metric_unit="token",
        )
