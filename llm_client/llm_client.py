"""Refactored LLM Client using Strategy Pattern with Providers."""

import os
import sys
from typing import Literal

from dotenv import load_dotenv

from .base_provider import BaseProvider
from .provider_factory import ProviderFactory


class LLMClient:
    """Universal client for interacting with various LLM providers.

    This client uses a strategy pattern with provider classes to handle
    different LLM APIs (OpenAI, Groq, Gemini, Ollama). It automatically
    detects available API keys or allows manual provider selection.

    Attributes:
        provider: The current LLM provider instance.
        api_choice: Name of the currently active API.
        llm: Name of the current model.
        temperature: Current sampling temperature.
        max_tokens: Current maximum tokens setting.

    Examples:
        >>> # Automatic API selection
        >>> client = LLMClient()
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response = client.chat_completion(messages)

        >>> # Manual provider selection
        >>> client = LLMClient(api_choice="gemini", llm="gemini-2.5-flash")

        >>> # Switch provider at runtime
        >>> client.switch_provider("openai", llm="gpt-4o")
    """

    def __init__(
        self,
        llm: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_choice: Literal["openai", "groq", "gemini", "ollama"] | None = None,
        secrets_path: str = "secrets.env",
        keep_alive: str = "5m",
    ) -> None:
        """Initialize the LLM Client.

        Args:
            llm: Model name. If None, uses provider's default.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            api_choice: Explicit API choice. If None, auto-selects.
            secrets_path: Path to secrets.env file.
            keep_alive: Ollama-specific keep-alive duration.

        Examples:
            >>> client = LLMClient(llm="gpt-4o", temperature=0.5)
            >>> client = LLMClient(api_choice="gemini")
        """
        # Load environment variables
        if os.path.exists(secrets_path):
            load_dotenv(secrets_path)

        # Load API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        # Try loading from Google Colab userdata
        self._load_colab_secrets()

        # Store configuration
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive
        self._user_specified_llm = llm

        # Create provider using factory
        self.provider: BaseProvider = ProviderFactory.create_provider(
            api_choice=api_choice,
            llm=llm,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=self.openai_api_key,
            groq_api_key=self.groq_api_key,
            gemini_api_key=self.gemini_api_key,
            keep_alive=keep_alive,
        )

        # Store current API choice (infer from provider class)
        self.api_choice = self._get_api_choice_from_provider()

    def _load_colab_secrets(self) -> None:
        """Load API keys from Google Colab userdata if available."""
        if "google.colab" not in sys.modules and "COLAB_GPU" not in os.environ:
            return

        try:
            from google.colab import userdata

            # Try loading each key individually
            if not self.openai_api_key:
                try:
                    self.openai_api_key = userdata.get("OPENAI_API_KEY")
                except Exception as e:
                    print(e)

            if not self.groq_api_key:
                try:
                    self.groq_api_key = userdata.get("GROQ_API_KEY")
                except Exception as e:
                    print(e)

            if not self.gemini_api_key:
                try:
                    self.gemini_api_key = userdata.get("GEMINI_API_KEY")
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)

    def _get_api_choice_from_provider(self) -> str:
        """Infer API choice from provider class name.

        Returns:
            API name as string.
        """
        provider_class_name = self.provider.__class__.__name__.lower()
        if "openai" in provider_class_name:
            return "openai"
        elif "groq" in provider_class_name:
            return "groq"
        elif "gemini" in provider_class_name:
            return "gemini"
        elif "ollama" in provider_class_name:
            return "ollama"
        return "unknown"

    def switch_provider(
        self,
        api_choice: Literal["openai", "groq", "gemini", "ollama"],
        llm: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Switch to a different LLM provider at runtime.

        This allows changing providers without creating a new client instance.
        Useful for fallback strategies, cost optimization, or A/B testing.

        Args:
            api_choice: Target API to switch to.
            llm: Optional new model name. If None, uses provider default.
            temperature: Optional new temperature. If None, keeps current.
            max_tokens: Optional new max_tokens. If None, keeps current.

        Raises:
            ValueError: If api_choice is invalid.
            RuntimeError: If API key for chosen provider is missing.

        Examples:
            >>> client = LLMClient(api_choice="openai")
            >>> client.switch_provider("gemini", llm="gemini-2.5-flash")
            >>> client.switch_provider("groq", temperature=0.3)
        """
        # Update parameters if provided
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens

        # Update user-specified model
        self._user_specified_llm = llm

        # Create new provider
        self.provider = ProviderFactory.create_provider(
            api_choice=api_choice,
            llm=llm,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=self.openai_api_key,
            groq_api_key=self.groq_api_key,
            gemini_api_key=self.gemini_api_key,
            keep_alive=self.keep_alive,
        )

        # Update API choice
        self.api_choice = api_choice.lower()

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        """Execute a chat completion using the current provider.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If the provider call fails.

        Examples:
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "Explain AI."}
            ... ]
            >>> response = client.chat_completion(messages)
        """
        return self.provider.chat_completion(messages)

    @property
    def llm(self) -> str:
        """Get the current model name.

        Returns:
            Name of the current model.
        """
        return self.provider.llm

    @property
    def client(self):
        """Get the underlying API client (for backward compatibility).

        Returns:
            The provider's client instance.
        """
        return self.provider.client

    def __repr__(self) -> str:
        """Return string representation of the client.

        Returns:
            String with client configuration info.
        """
        return (
            f"LLMClient(api={self.api_choice}, model={self.llm}, "
            f"temperature={self.temperature})"
        )
