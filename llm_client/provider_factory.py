"""Factory for creating LLM provider instances."""

import os
from typing import Literal

from .base_provider import BaseProvider
from .providers import GeminiProvider, GroqProvider, OllamaProvider, OpenAIProvider


class ProviderFactory:
    """Factory class for creating LLM provider instances.

    This factory handles the creation and configuration of different
    LLM providers based on the requested API choice and available API keys.
    """

    _provider_classes = {
        "openai": OpenAIProvider,
        "groq": GroqProvider,
        "gemini": GeminiProvider,
        "ollama": OllamaProvider,
    }

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
        keep_alive: str = "5m",
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
            keep_alive: Ollama keep-alive duration.

        Returns:
            Configured provider instance.

        Raises:
            ValueError: If api_choice is invalid.
            RuntimeError: If no API key is found and not using Ollama.
        """
        # Auto-select API if not specified
        if api_choice is None:
            api_choice = cls._auto_select_api(openai_api_key, groq_api_key, gemini_api_key)

        # Validate API choice
        api_choice = api_choice.lower()
        if api_choice not in cls._provider_classes:
            raise ValueError(
                f"Invalid api_choice: {api_choice}. "
                f"Must be one of {list(cls._provider_classes.keys())}"
            )

        # Get provider class
        provider_class = cls._provider_classes[api_choice]

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

        # Create and return provider
        return provider_class(llm=llm, temperature=temperature, max_tokens=max_tokens, **kwargs)

    @staticmethod
    def _auto_select_api(
        openai_api_key: str | None,
        groq_api_key: str | None,
        gemini_api_key: str | None,
    ) -> str:
        """Auto-select API based on available keys.

        Priority: OpenAI > Groq > Gemini > Ollama

        Args:
            openai_api_key: OpenAI API key.
            groq_api_key: Groq API key.
            gemini_api_key: Gemini API key.

        Returns:
            Selected API name as string.

        Raises:
            RuntimeError: If running in Colab without API keys.
        """
        import sys

        if openai_api_key:
            return "openai"
        elif groq_api_key:
            return "groq"
        elif gemini_api_key:
            return "gemini"
        else:
            # Check if in Google Colab - if so, require API key
            if "google.colab" in sys.modules or "COLAB_GPU" in os.environ:
                raise RuntimeError(
                    "Kein API-Key gefunden. Bitte OPENAI_API_KEY, GROQ_API_KEY "
                    "oder GEMINI_API_KEY in Colab-Umgebung setzen."
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
