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
    from ollama import Client

    OLLAMA_AVAILABLE = True
except ImportError:
    Client = None  # type: ignore
    OLLAMA_AVAILABLE = False


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

    def _chat_completion_with_tools_impl(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Execute chat completion with tools using OpenAI."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        kwargs = {
            "model": self.llm,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": tools,
        }

        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        return {
            "content": choice.message.content,
            "tool_calls": (
                [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in (choice.message.tool_calls or [])
                ]
                if choice.message.tool_calls
                else None
            ),
        }

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

    def _chat_completion_with_tools_impl(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Execute chat completion with tools using OpenAI."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        kwargs = {
            "model": self.llm,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": tools,
        }

        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        return {
            "content": choice.message.content,
            "tool_calls": (
                [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in (choice.message.tool_calls or [])
                ]
                if choice.message.tool_calls
                else None
            ),
        }

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

    def _chat_completion_with_tools_impl(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Execute chat completion with tools using OpenAI."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        kwargs = {
            "model": self.llm,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": tools,
        }

        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        return {
            "content": choice.message.content,
            "tool_calls": (
                [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in (choice.message.tool_calls or [])
                ]
                if choice.message.tool_calls
                else None
            ),
        }

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
    """Provider for Ollama API with local and cloud support.

    Supports both:
    - Local Ollama instance (default)
    - Ollama Cloud API (requires API key)

    Examples:
        >>> # Local Ollama
        >>> provider = OllamaProvider(llm="llama3.2:1b")

        >>> # Ollama Cloud
        >>> provider = OllamaProvider(
        ...     llm="gpt-oss:120b-cloud",
        ...     api_key="your_api_key",
        ...     use_cloud=True
        ... )
    """

    def __init__(
        self,
        llm: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        keep_alive: str = "5m",
        use_cloud: bool = False,
        host: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Ollama provider.

        Args:
            llm: Model name (use `-cloud` suffix for cloud models).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            keep_alive: How long to keep model in memory (local only).
            use_cloud: If True, use Ollama Cloud API.
            host: Custom Ollama host URL. If None, uses default.
            **kwargs: Additional parameters including 'api_key' for cloud.
        """
        self.keep_alive = keep_alive
        self.use_cloud = use_cloud or llm.endswith("-cloud")
        self.host = host
        self._api_key = kwargs.get("api_key")

        # Auto-detect cloud mode from model name
        if llm.endswith("-cloud") and not use_cloud:
            self.use_cloud = True

        super().__init__(llm, temperature, max_tokens, **kwargs)

    def _initialize_client(self, **kwargs: Any) -> None:
        """Initialize Ollama client (local or cloud).

        Args:
            **kwargs: May contain 'api_key' for Ollama Cloud.

        Raises:
            ProviderNotAvailableError: If ollama package is not installed.
            APIKeyNotFoundError: If cloud mode but API key is missing.
        """
        if not self.is_available():
            raise ProviderNotAvailableError("ollama", "ollama")

        if self.use_cloud:
            # Ollama Cloud mode
            api_key = kwargs.get("api_key") or self._api_key
            if not api_key:
                raise APIKeyNotFoundError("ollama_cloud", "OLLAMA_API_KEY")

            # Create client with cloud settings
            self.client = Client(
                host=self.host or "https://ollama.com",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        else:
            # Local Ollama mode
            if self.host:
                self.client = Client(host=self.host)
            else:
                # Use default local client
                self.client = Client()

    def _chat_completion_impl(self, messages: list[dict[str, str]]) -> str:
        """Execute chat completion with Ollama (local or cloud).

        Args:
            messages: List of message dictionaries.

        Returns:
            Generated text response.

        Raises:
            ProviderNotAvailableError: If ollama package is not available.
        """
        if not self.is_available():
            raise ProviderNotAvailableError("ollama", "ollama")

        options = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }

        # Add local-specific options
        if not self.use_cloud:
            options.update(
                {
                    "repeat_penalty": 1.2,
                    "top_k": 10,
                    "top_p": 0.5,
                }
            )

        kwargs = {
            "model": self.llm,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        # Add keep_alive for local mode only
        if not self.use_cloud:
            kwargs["keep_alive"] = self.keep_alive

        response = self.client.chat(**kwargs)
        return response["message"]["content"]

    def _chat_completion_with_tools_impl(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Tool calling support for Ollama."""
        if not self.is_available():
            raise ProviderNotAvailableError("ollama", "ollama")

        # Tool support is experimental in Ollama
        raise NotImplementedError(
            "Tool calling support in Ollama is experimental. "
            "Please check Ollama documentation for current status."
        )

    def _chat_completion_stream_impl(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream chat completion with Ollama (local or cloud).

        Args:
            messages: List of message dictionaries.

        Yields:
            Response text chunks as they arrive.

        Raises:
            ProviderNotAvailableError: If ollama package is not available.
        """
        if not self.is_available():
            raise ProviderNotAvailableError("ollama", "ollama")

        options = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }

        # Add local-specific options
        if not self.use_cloud:
            options.update(
                {
                    "repeat_penalty": 1.2,
                    "top_k": 10,
                    "top_p": 0.5,
                }
            )

        kwargs = {
            "model": self.llm,
            "messages": messages,
            "stream": True,
            "options": options,
        }

        # Add keep_alive for local mode only
        if not self.use_cloud:
            kwargs["keep_alive"] = self.keep_alive

        stream = self.client.chat(**kwargs)

        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]

    @staticmethod
    def get_default_model() -> str:
        """Get default Ollama model.

        Returns:
            Default model name (local model).
        """
        return "llama3.2:1b"

    @staticmethod
    def is_available() -> bool:
        """Check if Ollama package is available.

        Returns:
            True if ollama package is installed.
        """
        return OLLAMA_AVAILABLE

    def __repr__(self) -> str:
        """Return string representation of the provider.

        Returns:
            String with provider info.
        """
        mode = "cloud" if self.use_cloud else "local"
        return (
            f"OllamaProvider(model={self.llm}, " f"temperature={self.temperature}, " f"mode={mode})"
        )
