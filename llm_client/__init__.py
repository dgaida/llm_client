"""
llm_client
==========

A universal interface for LLM access using a strategy pattern with providers.

This package provides the `LLMClient` class that automatically detects
available APIs (based on environment variables) and uses the appropriate
provider for chat completions.

Main classes:
- LLMClient: Main client interface with streaming support
- BaseProvider: Abstract base class for providers
- ProviderFactory: Factory for creating provider instances

Provider implementations:
- OpenAIProvider: For OpenAI API
- GroqProvider: For Groq API
- GeminiProvider: For Google Gemini API
- OllamaProvider: For local Ollama API

Custom exceptions:
- LLMClientError: Base exception for all client errors
- APIKeyNotFoundError: Raised when API key is missing
- ProviderNotAvailableError: Raised when provider package not installed
- InvalidProviderError: Raised when invalid provider name specified
- ChatCompletionError: Raised when chat completion fails
- StreamingNotSupportedError: Raised when streaming not supported
"""

from .base_provider import BaseProvider
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

__all__ = [
    # Main classes
    "LLMClient",
    "BaseProvider",
    "ProviderFactory",
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
]

# Optional import of the adapter
try:
    from .adapter import LLMClientAdapter

    __all__.append("LLMClientAdapter")
except ImportError:
    # llama_index not installed - Adapter not available
    pass

__version__ = "0.2.0"
__author__ = "Daniel Gaida"
__license__ = "MIT"
