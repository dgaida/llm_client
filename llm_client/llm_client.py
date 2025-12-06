"""LLM Client with token counting, async support, and config file loading."""

import os
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from .base_provider import BaseProvider
from .config import LLMConfig
from .provider_factory import ProviderFactory
from .token_counter import TokenCounter


class LLMClient:
    """Universal client for interacting with various LLM providers.

    This client uses a strategy pattern with provider classes to handle
    different LLM APIs (OpenAI, Groq, Gemini, Ollama). It supports:
    - Automatic API detection or manual selection
    - Token counting with tiktoken
    - Async operations
    - Configuration file loading

    Attributes:
        provider: The current LLM provider instance.
        api_choice: Name of the currently active API.
        llm: Name of the current model.
        temperature: Current sampling temperature.
        max_tokens: Current maximum tokens setting.
        token_counter: TokenCounter instance for counting tokens.

    Examples:
        >>> # Automatic API selection
        >>> client = LLMClient()
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response = client.chat_completion(messages)

        >>> # From config file
        >>> client = LLMClient.from_config("llm_config.yaml")

        >>> # Count tokens
        >>> token_count = client.count_tokens(messages)

        >>> # Async usage
        >>> async_client = LLMClient(use_async=True)
        >>> response = await async_client.achat_completion(messages)
    """

    def __init__(
        self,
        llm: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_choice: Literal["openai", "groq", "gemini", "ollama"] | None = None,
        secrets_path: str = "secrets.env",
        keep_alive: str = "5m",
        use_async: bool = False,
    ) -> None:
        """Initialize the LLM Client.

        Args:
            llm: Model name. If None, uses provider's default.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            api_choice: Explicit API choice. If None, auto-selects.
            secrets_path: Path to secrets.env file.
            keep_alive: Ollama-specific keep-alive duration.
            use_async: If True, use async providers.

        Examples:
            >>> client = LLMClient(llm="gpt-4o", temperature=0.5)
            >>> client = LLMClient(api_choice="gemini")
            >>> async_client = LLMClient(use_async=True)
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
        self.use_async = use_async

        # Initialize token counter
        self.token_counter = TokenCounter()

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
            use_async=use_async,
        )

        # Store current API choice
        self.api_choice = self._get_api_choice_from_provider()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        provider: str | None = None,
        secrets_path: str = "secrets.env",
        use_async: bool = False,
    ) -> "LLMClient":
        """Create LLMClient from configuration file.

        Args:
            config_path: Path to YAML or JSON configuration file.
            provider: Provider to use. If None, uses default from config.
            secrets_path: Path to secrets.env file.
            use_async: If True, use async providers.

        Returns:
            Configured LLMClient instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If configuration is invalid.

        Examples:
            >>> # Use default provider from config
            >>> client = LLMClient.from_config("llm_config.yaml")

            >>> # Use specific provider
            >>> client = LLMClient.from_config("llm_config.yaml", provider="groq")

            >>> # Async client
            >>> client = LLMClient.from_config("llm_config.yaml", use_async=True)
        """
        # Load configuration
        config = LLMConfig.from_file(config_path)

        # Validate configuration
        is_valid, errors = config.validate()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

        # Determine which provider to use
        provider_name = provider or config.default_provider
        provider_config = config.get_provider_config(provider_name)

        # Extract parameters
        llm = provider_config.get("model")
        temperature = provider_config.get("temperature", 0.7)
        max_tokens = provider_config.get("max_tokens", 512)
        keep_alive = provider_config.get("keep_alive", "5m")

        # Create client
        return cls(
            llm=llm,
            temperature=temperature,
            max_tokens=max_tokens,
            api_choice=provider_name,
            secrets_path=secrets_path,
            keep_alive=keep_alive,
            use_async=use_async,
        )

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
            InvalidProviderError: If api_choice is invalid.
            APIKeyNotFoundError: If API key for chosen provider is missing.
            ProviderNotAvailableError: If provider package is not installed.

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
            use_async=self.use_async,
        )

        # Update API choice
        self.api_choice = api_choice.lower()

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        """Execute a chat completion using the current provider.

        This method includes automatic retry logic with exponential backoff
        to handle transient API failures.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            Generated text response.

        Raises:
            ChatCompletionError: If the provider call fails after retries.

        Examples:
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "Explain AI."}
            ... ]
            >>> response = client.chat_completion(messages)
        """
        return self.provider.chat_completion(messages)

    def chat_completion_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream response tokens as they arrive from the LLM.

        This method returns an iterator that yields response tokens in real-time,
        enabling progressive display of the response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Yields:
            Individual tokens or chunks of the response text.

        Raises:
            StreamingNotSupportedError: If streaming is not supported.
            ChatCompletionError: If the streaming API call fails.

        Examples:
            >>> messages = [{"role": "user", "content": "Tell me a story"}]
            >>> for chunk in client.chat_completion_stream(messages):
            ...     print(chunk, end="", flush=True)
            >>> print()  # New line after streaming completes
        """
        return self.provider.chat_completion_stream(messages)

    async def achat_completion(self, messages: list[dict[str, str]]) -> str:
        """Execute async chat completion.

        Args:
            messages: List of message dicts.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If provider doesn't support async.

        Examples:
            >>> response = await client.achat_completion(messages)
        """
        if not hasattr(self.provider, "achat_completion"):
            raise RuntimeError(
                f"{self.provider.__class__.__name__} does not support async. "
                f"Create client with use_async=True"
            )
        return await self.provider.achat_completion(messages)

    async def achat_completion_stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Stream response tokens asynchronously.

        Args:
            messages: List of message dicts.

        Yields:
            Individual tokens or chunks.

        Raises:
            RuntimeError: If provider doesn't support async streaming.

        Examples:
            >>> async for chunk in client.achat_completion_stream(messages):
            ...     print(chunk, end="", flush=True)
        """
        if not hasattr(self.provider, "achat_completion_stream"):
            raise RuntimeError(
                f"{self.provider.__class__.__name__} does not support async streaming"
            )
        async for chunk in self.provider.achat_completion_stream(messages):
            yield chunk

    def count_tokens(self, messages: list[dict[str, str]], model: str | None = None) -> int:
        """Count tokens in messages using tiktoken.

        Args:
            messages: List of message dicts to count tokens for.
            model: Model name for encoding. If None, uses current model.

        Returns:
            Total token count.

        Examples:
            >>> messages = [{"role": "user", "content": "Hello world"}]
            >>> token_count = client.count_tokens(messages)
            >>> print(f"Tokens: {token_count}")
        """
        model_name = model or self.llm
        return self.token_counter.count_tokens(messages, model=model_name)

    def count_string_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens in a string.

        Args:
            text: Text to count tokens for.
            model: Model name. If None, uses current model.

        Returns:
            Token count.

        Examples:
            >>> token_count = client.count_string_tokens("Hello world!")
        """
        model_name = model or self.llm
        return self.token_counter.count_string_tokens(text, model=model_name)

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
        async_suffix = " (async)" if self.use_async else ""
        return (
            f"LLMClient(api={self.api_choice}, model={self.llm}, "
            f"temperature={self.temperature}{async_suffix})"
        )
