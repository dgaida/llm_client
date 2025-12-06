"""
llm_client
==========

A universal interface for LLM access using a strategy pattern with providers.

This package provides the `LLMClient` class that automatically detects
available APIs (based on environment variables) and uses the appropriate
provider for chat completions.

Main classes:
- LLMClient: Main client interface
- BaseProvider: Abstract base class for providers
- ProviderFactory: Factory for creating provider instances

Provider implementations:
- OpenAIProvider: For OpenAI API
- GroqProvider: For Groq API
- GeminiProvider: For Google Gemini API
- OllamaProvider: For local Ollama API
"""

from .base_provider import BaseProvider
from .llm_client import LLMClient
from .provider_factory import ProviderFactory
from .providers import GeminiProvider, GroqProvider, OllamaProvider, OpenAIProvider

__all__ = [
    "LLMClient",
    "BaseProvider",
    "ProviderFactory",
    "OpenAIProvider",
    "GroqProvider",
    "GeminiProvider",
    "OllamaProvider",
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
