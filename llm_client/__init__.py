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

Logging configuration:
- setup_logging: Configure logging for the package
- enable_logging: Enable logging at specified level
- disable_logging: Disable all logging

New in v0.3.0:
- Token counting with tiktoken
- Full async/await support
- YAML/JSON configuration files
- Ollama Cloud support
- Comprehensive logging
"""

from .config import LLMConfig, create_default_config, generate_config_template
from .exceptions import (
    APIKeyNotFoundError,
    ChatCompletionError,
    FileUploadNotSupportedError,
    InvalidProviderError,
    LLMClientError,
    ProviderNotAvailableError,
    StreamingNotSupportedError,
)
from .llm_client import LLMClient
from .providers.base_provider import BaseProvider
from .providers.provider_factory import ProviderFactory
from .providers.providers import GeminiProvider, GroqProvider, OllamaProvider, OpenAIProvider
from .utils.file_utils import detect_file_type, validate_file_for_provider
from .utils.logging_config import disable_logging, enable_logging, setup_logging
from .utils.token_counter import TokenCounter

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
    "FileUploadNotSupportedError",
    "StreamingNotSupportedError",
    # Config utilities
    "create_default_config",
    "generate_config_template",
    # Utils
    "detect_file_type",
    "validate_file_for_provider",
    # Logging
    "setup_logging",
    "enable_logging",
    "disable_logging",
]

# Optional async providers
try:
    from .providers.async_providers import (
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
    from .providers.adapter import LLMClientAdapter

    __all__.append("LLMClientAdapter")
except ImportError:
    # llama_index not installed - Adapter not available
    pass

__version__ = "0.4.7"
__author__ = "Daniel Gaida"
__license__ = "MIT"
