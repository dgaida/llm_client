"""Concrete implementations of LLM providers."""

from typing import Any

from .base_provider import BaseProvider

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
    """Provider for OpenAI API."""

    def _initialize_client(self, **kwargs: Any) -> None:
        """Initialize OpenAI client.

        Args:
            **kwargs: Must contain 'api_key' for OpenAI authentication.

        Raises:
            RuntimeError: If OpenAI package is not installed or API key is missing.
        """
        if not self.is_available():
            raise RuntimeError("OpenAI package not available. Install with: pip install openai")

        api_key = kwargs.get("api_key")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found. Please set it in environment.")

        self.client = OpenAI(api_key=api_key)

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
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
    """Provider for Groq API."""

    def _initialize_client(self, **kwargs: Any) -> None:
        """Initialize Groq client.

        Args:
            **kwargs: Must contain 'api_key' for Groq authentication.

        Raises:
            RuntimeError: If Groq package is not installed or API key is missing.
        """
        if not self.is_available():
            raise RuntimeError("Groq package not available. Install with: pip install groq")

        api_key = kwargs.get("api_key")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found. Please set it in environment.")

        self.client = Groq(api_key=api_key)

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
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
            RuntimeError: If OpenAI package is not installed or API key is missing.
        """
        if not self.is_available():
            raise RuntimeError(
                "OpenAI package required for Gemini. Install with: pip install openai"
            )

        api_key = kwargs.get("api_key")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found. Please set it in environment.")

        # Use OpenAI client with Gemini's base URL
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
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
    """Provider for local Ollama API."""

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
            RuntimeError: If ollama package is not installed.
        """
        if not self.is_available():
            raise RuntimeError("Ollama package not available. Install with: pip install ollama")
        # Ollama doesn't need a client object
        self.client = None

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        """Execute chat completion with Ollama.

        Args:
            messages: List of message dictionaries.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If ollama package is not available.
        """
        if not self.is_available():
            raise RuntimeError(
                "Ollama Python package not available. "
                "Please install it via `pip install ollama`."
            )

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
