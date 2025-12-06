"""Concrete implementations of LLM providers with streaming support."""

from collections.abc import Iterator
from typing import Any

from .base_provider import BaseProvider
from .exceptions import APIKeyNotFoundError, ProviderNotAvailableError

# Optional imports - providers check availability
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI API with streaming support."""

    def _initialize_client(self, **kwargs: Any) -> None:
        """Initialize OpenAI client.

        Args:
            **kwargs: Must contain 'api_key' for OpenAI authentication.

        Raises:
            ProviderNotAvailableError: If OpenAI package is not installed.
            APIKeyNotFoundError: If API key is missing.
        """
        if not self.is_available():
            raise ProviderNotAvailableError("openai", "openai")

        api_key = kwargs.get("api_key")
        if not api_key:
            raise APIKeyNotFoundError("openai", "OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key)

    def _chat_completion_impl(self, messages: list[dict[str, str]]) -> str:
        """Execute chat completion with OpenAI.

        Args:
            messages: List of message dictionaries.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If client is not initialized.
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        response = self.client.chat.completions.create(
            model=self.llm,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def _chat_completion_stream_impl(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream chat completion with OpenAI.

        Args:
            messages: List of message dictionaries.

        Yields:
            Response text chunks as they arrive.

        Raises:
            RuntimeError: If client is not initialized.
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        stream = self.client.chat.completions.create(
            model=self.llm,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    @staticmethod
    def get_default_model() -> str:
        """Get default OpenAI model.

        Returns:
            Default model name.
        """
        return "gpt-4o-mini"

    @staticmethod
    def is_available() -> bool:
        """Check if OpenAI package is available.

        Returns:
            True if openai package is installed.
        """
        return OpenAI is not None


class GroqProvider(BaseProvider):
    """Provider for Groq API with streaming support."""

    def _initialize_client(self, **kwargs: Any) -> None:
        """Initialize Groq client.

        Args:
            **kwargs: Must contain 'api_key' for Groq authentication.

        Raises:
            ProviderNotAvailableError: If Groq package is not installed.
            APIKeyNotFoundError: If API key is missing.
        """
        if not self.is_available():
            raise ProviderNotAvailableError("groq", "groq")

        api_key = kwargs.get("api_key")
        if not api_key:
            raise APIKeyNotFoundError("groq", "GROQ_API_KEY")

        self.client = Groq(api_key=api_key)

    def _chat_completion_impl(self, messages: list[dict[str, str]]) -> str:
        """Execute chat completion with Groq.

        Args:
            messages: List of message dictionaries.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If client is not initialized.
        """
        if not self.client:
            raise RuntimeError("Groq client not initialized")

        response = self.client.chat.completions.create(
            model=self.llm,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def _chat_completion_stream_impl(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream chat completion with Groq.

        Args:
            messages: List of message dictionaries.

        Yields:
            Response text chunks as they arrive.

        Raises:
            RuntimeError: If client is not initialized.
        """
        if not self.client:
            raise RuntimeError("Groq client not initialized")

        stream = self.client.chat.completions.create(
            model=self.llm,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    @staticmethod
    def get_default_model() -> str:
        """Get default Groq model.

        Returns:
            Default model name.
        """
        return "moonshotai/kimi-k2-instruct-0905"

    @staticmethod
    def is_available() -> bool:
        """Check if Groq package is available.

        Returns:
            True if groq package is installed.
        """
        return Groq is not None


class GeminiProvider(BaseProvider):
    """Provider for Google Gemini API via OpenAI compatibility mode."""

    def _initialize_client(self, **kwargs: Any) -> None:
        """Initialize Gemini client using OpenAI compatibility.

        Args:
            **kwargs: Must contain 'api_key' for Gemini authentication.

        Raises:
            ProviderNotAvailableError: If OpenAI package is not installed.
            APIKeyNotFoundError: If API key is missing.
        """
        if not self.is_available():
            raise ProviderNotAvailableError("gemini", "openai")

        api_key = kwargs.get("api_key")
        if not api_key:
            raise APIKeyNotFoundError("gemini", "GEMINI_API_KEY")

        # Use OpenAI client with Gemini's base URL
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def _chat_completion_impl(self, messages: list[dict[str, str]]) -> str:
        """Execute chat completion with Gemini.

        Args:
            messages: List of message dictionaries.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If client is not initialized.
        """
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        response = self.client.chat.completions.create(
            model=self.llm,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def _chat_completion_stream_impl(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream chat completion with Gemini.

        Args:
            messages: List of message dictionaries.

        Yields:
            Response text chunks as they arrive.

        Raises:
            RuntimeError: If client is not initialized.
        """
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        stream = self.client.chat.completions.create(
            model=self.llm,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    @staticmethod
    def get_default_model() -> str:
        """Get default Gemini model.

        Returns:
            Default model name.
        """
        return "gemini-2.0-flash-exp"

    @staticmethod
    def is_available() -> bool:
        """Check if OpenAI package (needed for Gemini) is available.

        Returns:
            True if openai package is installed.
        """
        return OpenAI is not None


class OllamaProvider(BaseProvider):
    """Provider for local Ollama API with streaming support."""

    def __init__(
        self,
        llm: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        keep_alive: str = "5m",
        **kwargs: Any,
    ) -> None:
        """Initialize Ollama provider.

        Args:
            llm: Model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            keep_alive: How long to keep model in memory (Ollama-specific).
            **kwargs: Additional parameters.
        """
        self.keep_alive = keep_alive
        super().__init__(llm, temperature, max_tokens, **kwargs)

    def _initialize_client(self, **kwargs: Any) -> None:
        """Initialize Ollama (no client needed, uses module directly).

        Args:
            **kwargs: Unused for Ollama.

        Raises:
            ProviderNotAvailableError: If ollama package is not installed.
        """
        if not self.is_available():
            raise ProviderNotAvailableError("ollama", "ollama")
        # Ollama doesn't need a client object
        self.client = None

    def _chat_completion_impl(self, messages: list[dict[str, str]]) -> str:
        """Execute chat completion with Ollama.

        Args:
            messages: List of message dictionaries.

        Returns:
            Generated text response.

        Raises:
            ProviderNotAvailableError: If ollama package is not available.
        """
        if not self.is_available():
            raise ProviderNotAvailableError("ollama", "ollama")

        response = ollama.chat(
            model=self.llm,
            messages=messages,
            stream=False,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "repeat_penalty": 1.2,
                "top_k": 10,
                "top_p": 0.5,
            },
            keep_alive=self.keep_alive,
        )
        return response["message"]["content"]

    def _chat_completion_stream_impl(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream chat completion with Ollama.

        Args:
            messages: List of message dictionaries.

        Yields:
            Response text chunks as they arrive.

        Raises:
            ProviderNotAvailableError: If ollama package is not available.
        """
        if not self.is_available():
            raise ProviderNotAvailableError("ollama", "ollama")

        stream = ollama.chat(
            model=self.llm,
            messages=messages,
            stream=True,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "repeat_penalty": 1.2,
                "top_k": 10,
                "top_p": 0.5,
            },
            keep_alive=self.keep_alive,
        )

        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]

    @staticmethod
    def get_default_model() -> str:
        """Get default Ollama model.

        Returns:
            Default model name.
        """
        return "llama3.2:1b"

    @staticmethod
    def is_available() -> bool:
        """Check if Ollama package is available.

        Returns:
            True if ollama package is installed.
        """
        return ollama is not None
