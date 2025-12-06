"""
llm_client
==========

A universal interface for LLM access with streaming, async support, token counting,
and configuration file management.

Main classes:
- LLMClient: Main client interface with all features
- BaseProvider: Abstract base class for providers
- ProviderFactory: Factory for creating provider instances
- TokenCounter: Utility for counting tokens
- LLMConfig: Configuration file loader

Provider implementations:
- OpenAIProvider: For OpenAI API
- GroqProvider: For Groq API
- GeminiProvider: For Google Gemini API
- OllamaProvider: For local Ollama API

Async providers (optional):
- AsyncOpenAIProvider: Async OpenAI support
- AsyncGroqProvider: Async Groq support
- AsyncGeminiProvider: Async Gemini support

Custom exceptions:
- LLMClientError: Base exception
- APIKeyNotFoundError: Missing API key
- ProviderNotAvailableError: Package not installed
- InvalidProviderError: Invalid provider name
- ChatCompletionError: Chat completion failed
- StreamingNotSupportedError: Streaming not supported

New in v0.3.0:
- Token counting with tiktoken
- Full async/await support
- YAML/JSON configuration files
"""

from .base_provider import BaseProvider
from .config import LLMConfig, create_default_config, generate_config_template
from .exceptions import (
    APIKeyNotFoundError,
    ChatCompletionError,
    InvalidProviderError,
    LLMClientError,
    ProviderNotAvailableError,
    StreamingNotSupportedError,
)
from .llm_client import LLMClient
from .provider_factory import ProviderFactory
from .providers import GeminiProvider, GroqProvider, OllamaProvider, OpenAIProvider
from .token_counter import TokenCounter

__all__ = [
    # Main classes
    "LLMClient",
    "BaseProvider",
    "ProviderFactory",
    "TokenCounter",
    "LLMConfig",
    # Providers
    "OpenAIProvider",
    "GroqProvider",
    "GeminiProvider",
    "OllamaProvider",
    # Exceptions
    "LLMClientError",
    "APIKeyNotFoundError",
    "ProviderNotAvailableError",
    "InvalidProviderError",
    "ChatCompletionError",
    "StreamingNotSupportedError",
    # Config utilities
    "create_default_config",
    "generate_config_template",
]

# Optional async providers
try:
    from .async_providers import (
        AsyncGeminiProvider,
        AsyncGroqProvider,
        AsyncOpenAIProvider,
    )

    __all__.extend(["AsyncOpenAIProvider", "AsyncGroqProvider", "AsyncGeminiProvider"])
except ImportError:
    # Async providers not available
    pass

# Optional llama-index adapter
try:
    from .adapter import LLMClientAdapter

    __all__.append("LLMClientAdapter")
except ImportError:
    # llama_index not installed - Adapter not available
    pass

__version__ = "0.3.0"
__author__ = "Daniel Gaida"
__license__ = "MIT"
