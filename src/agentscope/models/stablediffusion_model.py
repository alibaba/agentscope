# -*- coding: utf-8 -*-
"""Model wrapper for stable diffusion models."""
from abc import ABC
from typing import Any, Union, Sequence

try:
    import webuiapi
except ImportError:
    webuiapi = None

from . import ModelWrapperBase, ModelResponse
from ..message import Msg
from ..manager import FileManager
from ..utils.common import _convert_to_str


class StableDiffusionWrapperBase(ModelWrapperBase, ABC):
    """The base class for stable-diffusion model wrappers.

    To use SD-webui API, please
    1. First download stable-diffusion-webui from
    https://github.com/AUTOMATIC1111/stable-diffusion-webui and
    install it
    2. Move your checkpoint to 'models/Stable-diffusion' folder
    3. Start launch.py with the '--api --port=7862' parameter
    4. Install the 'webuiapi' package by 'pip install webuiapi'
    After that, you can use the SD-webui API and
    query the available parameters on the http://localhost:7862/docs page
    """

    model_type: str = "stable_diffusion"

    def __init__(
        self,
        config_name: str,
        generate_args: dict = None,
        options: dict = None,
        host: str = "127.0.0.1",
        port: int = 7862,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the SD-webui API client.

        Args:
            config_name (`str`):
                The name of the model config.
            generate_args (`dict`, default `None`):
                The extra keyword arguments used in SD api generation,
                e.g. `{"steps": 50}`.
            options (`dict`, default `None`):
                The keyword arguments to change the sd-webui settings
                such as model or CLIP skip, this changes will persist.
                e.g. `{"sd_model_checkpoint": "Anything-V3.0-pruned"}`.
            host (`str`, default `"127.0.0.1"`):
                The host of the stable-diffusion webui server.
            port (`int`, default `7862`):
                The port of the stable-diffusion webui server.
        """
        # Initialize the SD-webui API
        self.api = webuiapi.WebUIApi(host=host, port=port, **kwargs)
        self.generate_args = generate_args or {}

        # Set options if provided
        if options:
            self.api.set_options(options)

        # Get the default model name from the web-options
        model_name = (
            self.api.get_options()["sd_model_checkpoint"].split("[")[0].strip()
        )
        # Update the model name
        if self.generate_args.get("override_settings"):
            model_name = generate_args["override_settings"].get(
                "sd_model_checkpoint",
                model_name,
            )

        super().__init__(config_name=config_name, model_name=model_name)


class StableDiffusionImageSynthesisWrapper(StableDiffusionWrapperBase):
    """Stable Diffusion Text-to-Image (txt2img) API Wrapper"""

    model_type: str = "sd_txt2img"

    def __call__(
        self,
        prompt: str,
        save_local: bool = True,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Args:
            prompt (`str`):
                The prompt string to generate images from.
            save_local (`bool`, default `True`):
                Whether to save the generated images locally.
            **kwargs (`Any`):
                The keyword arguments to SD-webui txt2img API, e.g.
                `n_iter`, `steps`, `seed`, `width`, etc. Please refer to
                https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API
                or http://localhost:7862/docs
                for more detailed arguments.
        Returns:
            `ModelResponse`:
                A list of image local urls in image_urls field and the
                raw response in raw field.
        """
        # step1: prepare keyword arguments
        payload = {
            "prompt": prompt,
            **kwargs,
            **self.generate_args,
        }

        # step2: forward to generate response
        response = self.api.txt2img(**payload)

        # step3: save model invocation and update monitor
        self._save_model_invocation_and_update_monitor(
            payload=payload,
            response=response.json,
        )

        # step4: parse the response
        PIL_images = response.images

        file_manager = FileManager.get_instance()
        if save_local:
            # Save images
            image_urls = [file_manager.save_image(_) for _ in PIL_images]
            text = "Image saved to " + "\n".join(image_urls)
        else:
            image_urls = PIL_images
            text = ""  # Just a placeholder

        return ModelResponse(
            text=text,
            image_urls=image_urls,
            raw=response.json,
        )

    def _save_model_invocation_and_update_monitor(
        self,
        payload: dict,
        response: dict,
    ) -> None:
        """Save the model invocation and update the monitor accordingly.

        Args:
            kwargs (`dict`):
                The keyword arguments to the DashScope chat API.
            response (`dict`):
                The response object returned by the DashScope chat API.
        """
        self._save_model_invocation(
            arguments=payload,
            response=response,
        )

        session_parameters = response["parameters"]
        size = f"{session_parameters['width']}*{session_parameters['height']}"
        image_count = (
            session_parameters["batch_size"] * session_parameters["n_iter"]
        )

        self.monitor.update_image_tokens(
            model_name=self.model_name,
            image_count=image_count,
            resolution=size,
        )

    def format(self, *args: Union[Msg, Sequence[Msg]]) -> str:
        # This is a temporary implementation to focus on the prompt
        # on single-turn image generation by preserving only the system prompt
        # and the last user message. This logic might change in the future
        # to support more complex conversational scenarios
        if len(args) == 0:
            raise ValueError(
                "At least one message should be provided. An empty message "
                "list is not allowed.",
            )

        # Parse all information into a list of messages
        input_msgs = []
        for _ in args:
            if _ is None:
                continue
            if isinstance(_, Msg):
                input_msgs.append(_)
            elif isinstance(_, list) and all(isinstance(__, Msg) for __ in _):
                input_msgs.extend(_)
            else:
                raise TypeError(
                    f"The input should be a Msg object or a list "
                    f"of Msg objects, got {type(_)}.",
                )

        # record user message history as a list of strings
        user_messages = []
        sys_prompt = None
        for i, unit in enumerate(input_msgs):
            if i == 0 and unit.role == "system":
                # if system prompt is available, place it at the beginning
                sys_prompt = _convert_to_str(unit.content)
            elif unit.role == "user":
                # Merge user messages into a conversation history prompt
                user_messages.append(_convert_to_str(unit.content))
            else:
                continue

        content_components = []
        # Add system prompt at the beginning if provided
        if sys_prompt:
            content_components.append(sys_prompt)
        # Add the last user message if the user messages is not empty
        if len(user_messages) > 0:
            content_components.append(user_messages[-1])

        prompt = ",".join(content_components)

        return prompt
