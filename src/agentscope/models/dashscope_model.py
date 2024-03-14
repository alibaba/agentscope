# -*- coding: utf-8 -*-
"""Model wrapper for DashScope models"""
from http import HTTPStatus
from typing import Any, Union

try:
    import dashscope
except ModuleNotFoundError:
    dashscope = None

from loguru import logger

from .model import ModelWrapperBase, ModelResponse

from ..file_manager import file_manager
from ..utils.monitor import MonitorFactory
from ..utils.monitor import get_full_name
from ..constants import _DEFAULT_API_BUDGET


class DashScopeWrapper(ModelWrapperBase):
    """The model wrapper for DashScope API."""

    def __init__(
        self,
        config_name: str,
        model_name: str = None,
        api_key: str = None,
        generate_args: dict = None,
        budget: float = _DEFAULT_API_BUDGET,
        **kwargs: Any,
    ) -> None:
        """Initialize the DashScope wrapper.

        Args:
            config_name (`str`):
                The name of the model config.
            model_name (`str`, default `None`):
                The name of the model to use in DashScope API.
            api_key (`str`, default `None`):
                The API key for DashScope API.
            generate_args (`dict`, default `None`):
                The extra keyword arguments used in DashScope api generation,
                e.g. `temperature`, `seed`.
            budget (`float`, default `None`):
                The total budget using this model. Set to `None` means no
                limit.
        """
        if model_name is None:
            model_name = config_name
            logger.warning("model_name is not set, use config_name instead.")
        super().__init__(
            config_name=config_name,
            model_name=model_name,
            generate_args=generate_args,
            budget=budget,
            **kwargs,
        )
        if dashscope is None:
            raise ImportError(
                "Cannot find dashscope package in current python environment.",
            )

        self.model = model_name
        self.generate_args = generate_args or {}

        self.api_key = api_key
        dashscope.api_key = self.api_key
        self.max_length = None

        # Set monitor accordingly
        self.monitor = None
        self.budget = budget
        self._register_budget()
        self._register_default_metrics()

    def _register_budget(self) -> None:
        self.monitor = MonitorFactory.get_monitor()
        self.monitor.register_budget(
            model_name=self.model,
            value=self.budget,
            prefix=self.model,
        )

    def _register_default_metrics(self) -> None:
        """Register metrics to the monitor."""
        raise NotImplementedError(
            "The _register_default_metrics function is not Implemented.",
        )

    def _metric(self, metric_name: str) -> str:
        """Add the class name and model name as prefix to the metric name.

        Args:
            metric_name (`str`):
                The metric name.

        Returns:
            `str`: Metric name of this wrapper.
        """
        return get_full_name(name=metric_name, prefix=self.model)


class DashScopeChatWrapper(DashScopeWrapper):
    """The model wrapper for DashScope's chat API."""

    model_type: str = "dashscope_chat"

    deprecated_model_type: str = "tongyi_chat"

    def _register_default_metrics(self) -> None:
        # Set monitor accordingly
        # TODO: set quota to the following metrics
        self.monitor = MonitorFactory.get_monitor()
        self.monitor.register(
            self._metric("prompt_tokens"),
            metric_unit="token",
        )
        self.monitor.register(
            self._metric("completion_tokens"),
            metric_unit="token",
        )
        self.monitor.register(
            self._metric("total_tokens"),
            metric_unit="token",
        )

    def __call__(
        self,
        messages: list,
        **kwargs: Any,
    ) -> ModelResponse:
        """Processes a list of messages to construct a payload for the
        DashScope API call. It then makes a request to the DashScope API
        and returns the response. This method also updates monitoring
        metrics based on the API response.

        Each message in the 'messages' list can contain text content and
        optionally an 'image_urls' key. If 'image_urls' is provided,
        it is expected to be a list of strings representing URLs to images.
        These URLs will be transformed to a suitable format for the DashScope
        API, which might involve converting local file paths to data URIs.

        Args:
            messages (`list`):
                A list of messages to process.
            **kwargs (`Any`):
                The keyword arguments to DashScope chat completions API,
                e.g. `temperature`, `max_tokens`, `top_p`, etc. Please
                refer to
                https://help.aliyun.com/zh/dashscope/developer-reference/api-details
                for more detailed arguments.

        Returns:
            `ModelResponse`:
                The response text in text field, and the raw response in
                raw field.

        Note:
            `parse_func`, `fault_handler` and `max_retries` are reserved for
            `_response_parse_decorator` to parse and check the response
            generated by model wrapper. Their usages are listed as follows:
                - `parse_func` is a callable function used to parse and check
                the response generated by the model, which takes the response
                as input.
                - `max_retries` is the maximum number of retries when the
                `parse_func` raise an exception.
                - `fault_handler` is a callable function which is called
                when the response generated by the model is invalid after
                `max_retries` retries.
            The rule of roles in messages for DashScope is very rigid,
            for more details, please refer to
            https://help.aliyun.com/zh/dashscope/developer-reference/api-details
        """

        # step1: prepare keyword arguments
        kwargs = {**self.generate_args, **kwargs}

        # step2: checking messages
        if not all("role" in msg and "content" in msg for msg in messages):
            raise ValueError(
                "Each message in the 'messages' list must contain a 'role' "
                "and 'content' key for DashScope API.",
            )
        print(messages)
        # step3: forward to generate response
        response = dashscope.Generation.call(
            model=self.model,
            messages=messages,
            result_format="message",  # set the result to be "message" format.
            **kwargs,
        )

        if response.status_code != HTTPStatus.OK:
            error_msg = (
                f" Request id: {response.request_id},"
                f" Status code: {response.status_code},"
                f" error code: {response.code},"
                f" error message: {response.message}."
            )

            raise RuntimeError(error_msg)

        # step4: record the api invocation if needed
        self._save_model_invocation(
            arguments={
                "model": self.model,
                "messages": messages,
                **kwargs,
            },
            json_response=response,
        )

        # step5: update monitor accordingly
        try:
            self.monitor.update(
                {
                    "prompt_tokens": response.usage["input_tokens"],
                    "completion_tokens": response.usage["output_tokens"],
                    "total_tokens": response.usage["total_tokens"],
                },
                prefix=self.model,
            )
        except Exception as e:
            logger.error(e)

        # step6: return response
        return ModelResponse(
            text=response.output["choices"][0]["message"]["content"],
            raw=response,
        )


class DashScopeImageSynthesisWrapper(DashScopeWrapper):
    """The model wrapper for DashScope Image Synthesis API."""

    model_type: str = "dashscope_image_synthesis"

    def _register_default_metrics(self) -> None:
        # Set monitor accordingly
        # TODO: set quota to the following metrics
        self.monitor = MonitorFactory.get_monitor()
        self.monitor.register(
            self._metric("image_count"),
            metric_unit="image",
        )

    def __call__(
        self,
        prompt: str,
        save_local: bool = False,
        **kwargs: Any,
    ) -> ModelResponse:
        """
         Args:
             prompt (`str`):
                 The prompt string to generate images from.
             save_local: (`bool`, default `False`):
                 Whether to save the generated images locally, and replace
                 the returned image url with the local path.
             **kwargs (`Any`):
                 The keyword arguments to DashScope Image Synthesis API,
                 e.g. `n`, `size`, etc. Please refer to
                 https://help.aliyun.com/zh/dashscope/developer-reference/api-details-9
        for more detailed arguments.

         Returns:
             `ModelResponse`:
                 A list of image urls in image_urls field and the
                 raw response in raw field.

         Note:
             `parse_func`, `fault_handler` and `max_retries` are reserved
             for `_response_parse_decorator` to parse and check the
             response generated by model wrapper. Their usages are listed
             as follows:
                 - `parse_func` is a callable function used to parse and
                 check the response generated by the model, which takes
                 the response as input.
                 - `max_retries` is the maximum number of retries when the
                 `parse_func` raise an exception.
                 - `fault_handler` is a callable function which is called
                 when the response generated by the model is invalid after
                 `max_retries` retries.
        """
        # step1: prepare keyword arguments
        kwargs = {**self.generate_args, **kwargs}

        # step2: forward to generate response
        response = dashscope.ImageSynthesis.call(
            model=self.model,
            prompt=prompt,
            n=1,
            **kwargs,
        )
        if response.status_code != HTTPStatus.OK:
            error_msg = (
                f" Request id: {response.request_id},"
                f" Status code: {response.status_code},"
                f" error code: {response.code},"
                f" error message: {response.message}."
            )
            raise RuntimeError(error_msg)

        # step3: record the model api invocation if needed
        self._save_model_invocation(
            arguments={
                "model": self.model,
                "prompt": prompt,
                **kwargs,
            },
            json_response=response,
        )

        # step4: update monitor accordingly
        try:
            self.monitor.update(
                response.usage,
                prefix=self.model,
            )
        except Exception as e:
            logger.error(e)

        # step5: return response
        images = response["output"]["results"]
        # Get image urls as a list
        urls = [_["url"] for _ in images]

        if save_local:
            # Return local url if save_local is True
            urls = [file_manager.save_image(_) for _ in urls]
        return ModelResponse(image_urls=urls, raw=response)


class DashScopeTextEmbeddingWrapper(DashScopeWrapper):
    """The model wrapper for DashScope Text Embedding API."""

    model_type: str = "dashscope_text_embedding"

    def _register_default_metrics(self) -> None:
        # Set monitor accordingly
        # TODO: set quota to the following metrics
        self.monitor = MonitorFactory.get_monitor()
        self.monitor.register(
            self._metric("total_tokens"),
            metric_unit="token",
        )

    def __call__(
        self,
        texts: Union[list[str], str],
        **kwargs: Any,
    ) -> ModelResponse:
        """Embed the messages with DashScope Text Embedding API.

        Args:
            texts (`list[str]` or `str`):
                The messages used to embed.
            **kwargs (`Any`):
                The keyword arguments to DashScope Text Embedding API,
                e.g. `text_type`. Please refer to
                https://help.aliyun.com/zh/dashscope/developer-reference/api-details-15
                for more detailed arguments.

        Returns:
            `ModelResponse`:
                A list of embeddings in embedding field and the raw
                response in raw field.

        Note:
            `parse_func`, `fault_handler` and `max_retries` are reserved
            for `_response_parse_decorator` to parse and check the response
            generated by model wrapper. Their usages are listed as follows:
                - `parse_func` is a callable function used to parse and
                check the response generated by the model, which takes the
                response as input.
                - `max_retries` is the maximum number of retries when the
                `parse_func` raise an exception.
                - `fault_handler` is a callable function which is called
                when the response generated by the model is invalid after
                `max_retries` retries.
        """
        # step1: prepare keyword arguments
        kwargs = {**self.generate_args, **kwargs}

        # step2: forward to generate response
        response = dashscope.TextEmbedding.call(
            input=texts,
            model=self.model,
            **kwargs,
        )

        if response.status_code != HTTPStatus.OK:
            error_msg = (
                f" Request id: {response.request_id},"
                f" Status code: {response.status_code},"
                f" error code: {response.code},"
                f" error message: {response.message}."
            )
            raise RuntimeError(error_msg)

        # step3: record the model api invocation if needed
        self._save_model_invocation(
            arguments={
                "model": self.model,
                "input": texts,
                **kwargs,
            },
            json_response=response,
        )

        # step4: update monitor accordingly
        try:
            self.monitor.update(
                response.usage,
                prefix=self.model,
            )
        except Exception as e:
            logger.error(e)

        # step5: return response
        if len(response["output"]["embeddings"]) == 0:
            return ModelResponse(
                embedding=response["output"]["embedding"][0],
                raw=response,
            )
        else:
            return ModelResponse(
                embedding=[
                    _["embedding"] for _ in response["output"]["embeddings"]
                ],
                raw=response,
            )
