# -*- coding: utf-8 -*-
""" Import modules in models package."""

from .model import ModelWrapperBase
from .response import ModelResponse
from .post_model import (
    PostAPIModelWrapperBase,
    PostAPIChatWrapper,
)
from .openai_model import (
    OpenAIWrapperBase,
    OpenAIChatWrapper,
    OpenAIDALLEWrapper,
    OpenAIEmbeddingWrapper,
)
from .dashscope_model import (
    DashScopeChatWrapper,
    DashScopeImageSynthesisWrapper,
    DashScopeTextEmbeddingWrapper,
    DashScopeMultiModalWrapper,
)
from .ollama_model import (
    OllamaChatWrapper,
    OllamaEmbeddingWrapper,
    OllamaGenerationWrapper,
)
from .gemini_model import (
    GeminiChatWrapper,
    GeminiEmbeddingWrapper,
)
from .zhipu_model import (
    ZhipuAIChatWrapper,
    ZhipuAIEmbeddingWrapper,
)
from .litellm_model import (
    LiteLLMChatWrapper,
)
from .yi_model import (
    YiChatWrapper,
)
from .anthropic_model import AnthropicChatWrapper
from .stablediffusion_model import (
    StableDiffusionImageSynthesisWrapper,
)

_BUILD_IN_MODEL_WRAPPERS = [
    "PostAPIChatWrapper",
    "OpenAIChatWrapper",
    "OpenAIDALLEWrapper",
    "OpenAIEmbeddingWrapper",
    "DashScopeChatWrapper",
    "DashScopeImageSynthesisWrapper",
    "DashScopeTextEmbeddingWrapper",
    "DashScopeMultiModalWrapper",
    "OllamaChatWrapper",
    "OllamaEmbeddingWrapper",
    "OllamaGenerationWrapper",
    "GeminiChatWrapper",
    "GeminiEmbeddingWrapper",
    "ZhipuAIChatWrapper",
    "ZhipuAIEmbeddingWrapper",
    "LiteLLMChatWrapper",
    "YiChatWrapper",
    "AnthropicChatWrapper",
    "StableDiffusionImageSynthesisWrapper",
]

__all__ = [
    "ModelWrapperBase",
    "ModelResponse",
    "PostAPIModelWrapperBase",
    "PostAPIChatWrapper",
    "OpenAIWrapperBase",
    "OpenAIChatWrapper",
    "OpenAIDALLEWrapper",
    "OpenAIEmbeddingWrapper",
    "DashScopeChatWrapper",
    "DashScopeImageSynthesisWrapper",
    "DashScopeTextEmbeddingWrapper",
    "DashScopeMultiModalWrapper",
    "OllamaChatWrapper",
    "OllamaEmbeddingWrapper",
    "OllamaGenerationWrapper",
    "GeminiChatWrapper",
    "GeminiEmbeddingWrapper",
    "ZhipuAIChatWrapper",
    "ZhipuAIEmbeddingWrapper",
    "LiteLLMChatWrapper",
    "YiChatWrapper",
    "StableDiffusionImageSynthesisWrapper",
    "AnthropicChatWrapper",
]
