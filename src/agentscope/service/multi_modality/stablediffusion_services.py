# -*- coding: utf-8 -*-
"""Use StableDiffusion-webui API to generate images
"""
import os
from typing import Optional

from ...models import StableDiffusionImageSynthesisWrapper

from ...manager import FileManager
from ..service_response import (
    ServiceResponse,
    ServiceExecStatus,
)
from ...utils.common import (
    _get_timestamp,
    _generate_random_code,
)
from ...constants import _DEFAULT_IMAGE_NAME


def sd_text_to_image(
    prompt: str,
    n_iter: int = 1,
    width: int = 1024,
    height: int = 1024,
    options: dict = None,
    baseurl: str = None,
    save_dir: Optional[str] = None,
) -> ServiceResponse:
    """Generate image(s) based on the given prompt, and return image url(s).

    Args:
        prompt (`str`):
            The text prompt to generate image.
        n (`int`, defaults to `1`):
            The number of images to generate.
        width (`int`, defaults to `1024`):
            Width of the image.
        height (`int`, defaults to `1024`):
            Height of the image.
        options (`dict`, defaults to `None`):
            The options to override the sd-webui default settings.
            If not specified, will use the default settings.
        baseurl (`str`, defaults to `None`):
            The base url of the sd-webui.
        save_dir (`Optional[str]`, defaults to 'None'):
            The directory to save the generated images. If not specified,
            will return the web urls.

    Returns:
        ServiceResponse:
        A dictionary with two variables: `status` and`content`.
        If `status` is ServiceExecStatus.SUCCESS,
        the `content` is a dict with key 'fig_paths" and
        value is a list of the paths to the generated images.

    Example:

        .. code-block:: python

            prompt = "A beautiful sunset in the mountains"
            print(sd_text_to_image(prompt, 2))

    > {
    >     'status': 'SUCCESS',
    >     'content': {'image_urls': ['IMAGE_URL1', 'IMAGE_URL2']}
    > }

    """
    text2img = StableDiffusionImageSynthesisWrapper(
        config_name="sd-text-to-image-service",  # Just a placeholder
        baseurl=baseurl,
    )
    try:
        kwargs = {"n_iter": n_iter, "width": width, "height": height}
        if options:
            kwargs["override_settings"] = options

        res = text2img(prompt=prompt, save_local=False, **kwargs)
        images = res.image_urls

        # save images to save_dir
        if images is not None:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                urls_local = []
                # Obtain the image file names in the url
                for image in images:
                    image_name = _DEFAULT_IMAGE_NAME.format(
                        _get_timestamp(
                            "%Y%m%d-%H%M%S",
                        ),
                        _generate_random_code(),
                    )
                    image_path = os.path.abspath(
                        os.path.join(save_dir, image_name),
                    )
                    # Download the image
                    image.save(image_path)
                    urls_local.append(image_path)
                return ServiceResponse(
                    ServiceExecStatus.SUCCESS,
                    {"image_urls": urls_local},
                )
            else:
                # Return the default urls
                file_manager = FileManager.get_instance()
                urls = [file_manager.save_image(_) for _ in images]
                return ServiceResponse(
                    ServiceExecStatus.SUCCESS,
                    {"image_urls": urls},
                )
        else:
            return ServiceResponse(
                ServiceExecStatus.ERROR,
                "Error: Failed to generate images",
            )
    except Exception as e:
        return ServiceResponse(
            ServiceExecStatus.ERROR,
            str(e),
        )
