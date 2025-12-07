"""Factory for creating LLM provider instances with Ollama Cloud support."""

import os
from typing import Literal

from .base_provider import BaseProvider
from .exceptions import APIKeyNotFoundError, InvalidProviderError
from .providers import GeminiProvider, GroqProvider, OllamaProvider, OpenAIProvider


class ProviderFactory:
    """Factory class for creating LLM provider instances.

    This factory handles the creation and configuration of different
    LLM providers based on the requested API choice and available API keys.
    Supports both sync and async providers, including Ollama Cloud.
    """

    _provider_classes = {
        "openai": OpenAIProvider,
        "groq": GroqProvider,
        "gemini": GeminiProvider,
        "ollama": OllamaProvider,
    }

    _async_provider_classes = {}  # Will be populated if async module available

    @classmethod
    def create_provider(
        cls,
        api_choice: Literal["openai", "groq", "gemini", "ollama"] | None = None,
        llm: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        openai_api_key: str | None = None,
        groq_api_key: str | None = None,
        gemini_api_key: str | None = None,
        ollama_api_key: str | None = None,
        keep_alive: str = "5m",
        use_async: bool = False,
        use_ollama_cloud: bool = False,
        ollama_host: str | None = None,
    ) -> BaseProvider:
        """Create a provider instance.

        Args:
            api_choice: Explicit API choice. If None, auto-selects based on keys.
            llm: Model name. If None, uses provider's default.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            openai_api_key: OpenAI API key.
            groq_api_key: Groq API key.
            gemini_api_key: Gemini API key.
            ollama_api_key: Ollama Cloud API key.
            keep_alive: Ollama keep-alive duration.
            use_async: If True, create async provider.
            use_ollama_cloud: If True, use Ollama Cloud instead of local.
            ollama_host: Custom Ollama host URL.

        Returns:
            Configured provider instance.

        Raises:
            InvalidProviderError: If api_choice is invalid.
            APIKeyNotFoundError: If required API key is missing.
            ProviderNotAvailableError: If required package is not installed.
        """
        # Lazy load async providers
        if use_async and not cls._async_provider_classes:
            cls._load_async_providers()

        # Auto-detect Ollama Cloud from model name
        if llm and llm.endswith("-cloud"):
            use_ollama_cloud = True
            if api_choice is None:
                api_choice = "ollama"

        # Auto-select API if not specified
        if api_choice is None:
            api_choice = cls._auto_select_api(
                openai_api_key, groq_api_key, gemini_api_key, ollama_api_key, use_ollama_cloud
            )

        # Validate API choice
        api_choice = api_choice.lower()

        # Select provider classes based on async flag
        provider_classes = cls._async_provider_classes if use_async else cls._provider_classes

        if api_choice not in provider_classes:
            valid_providers = list(cls._provider_classes.keys())
            raise InvalidProviderError(api_choice, valid_providers)

        # Get provider class
        provider_class = provider_classes[api_choice]

        # Get model name (use default if not specified)
        if llm is None:
            llm = provider_class.get_default_model()

        # Prepare kwargs based on provider type
        kwargs = {}
        if api_choice == "openai":
            kwargs["api_key"] = openai_api_key
        elif api_choice == "groq":
            kwargs["api_key"] = groq_api_key
        elif api_choice == "gemini":
            kwargs["api_key"] = gemini_api_key
        elif api_choice == "ollama":
            kwargs["keep_alive"] = keep_alive
            kwargs["use_cloud"] = use_ollama_cloud
            if ollama_host:
                kwargs["host"] = ollama_host
            if use_ollama_cloud and ollama_api_key:
                kwargs["api_key"] = ollama_api_key

        # Create and return provider
        return provider_class(llm=llm, temperature=temperature, max_tokens=max_tokens, **kwargs)

    @classmethod
    def _load_async_providers(cls) -> None:
        """Load async provider classes."""
        try:
            from .async_providers import (
                AsyncGeminiProvider,
                AsyncGroqProvider,
                AsyncOpenAIProvider,
            )

            cls._async_provider_classes = {
                "openai": AsyncOpenAIProvider,
                "groq": AsyncGroqProvider,
                "gemini": AsyncGeminiProvider,
                "ollama": OllamaProvider,  # Ollama doesn't have async yet
            }
        except ImportError:
            # Async providers not available
            cls._async_provider_classes = cls._provider_classes.copy()

    @staticmethod
    def _auto_select_api(
        openai_api_key: str | None,
        groq_api_key: str | None,
        gemini_api_key: str | None,
        ollama_api_key: str | None = None,
        use_ollama_cloud: bool = False,
    ) -> str:
        """Auto-select API based on available keys.

        Priority: OpenAI > Groq > Gemini > Ollama Cloud (if key) > Ollama Local

        Args:
            openai_api_key: OpenAI API key.
            groq_api_key: Groq API key.
            gemini_api_key: Gemini API key.
            ollama_api_key: Ollama Cloud API key.
            use_ollama_cloud: If True, prefer Ollama Cloud over local.

        Returns:
            Selected API name as string.

        Raises:
            APIKeyNotFoundError: If running in Colab without API keys.
        """
        import sys

        if openai_api_key:
            return "openai"
        elif groq_api_key:
            return "groq"
        elif gemini_api_key:
            return "gemini"
        elif use_ollama_cloud and ollama_api_key:
            return "ollama"
        else:
            # Check if in Google Colab - if so, require API key
            if "google.colab" in sys.modules or "COLAB_GPU" in os.environ:
                raise APIKeyNotFoundError(
                    "colab",
                    "OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, or OLLAMA_API_KEY",
                )
            return "ollama"

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available providers (where package is installed).

        Returns:
            List of provider names that are available.
        """
        available = []
        for name, provider_class in cls._provider_classes.items():
            if provider_class.is_available():
                available.append(name)
        return available

    @classmethod
    def is_provider_available(cls, provider_name: str) -> bool:
        """Check if a specific provider is available.

        Args:
            provider_name: Name of the provider to check.

        Returns:
            True if provider package is installed.
        """
        provider_class = cls._provider_classes.get(provider_name.lower())
        if provider_class is None:
            return False
        return provider_class.is_available()
