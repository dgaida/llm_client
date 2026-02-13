"""Base provider interface for LLM clients with streaming support."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from ..exceptions import ChatCompletionError


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    This class defines the interface that all LLM providers must implement.
    Each provider handles the specific API communication and response parsing
    for its respective service.

    Attributes:
        llm: Name of the model to use.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum number of tokens to generate.
        client: The underlying API client instance.
    """

    def __init__(
        self, llm: str, temperature: float = 0.7, max_tokens: int = 512, **kwargs: Any
    ) -> None:
        """Initialize the provider.

        Args:
            llm (str): Model name to use.
            temperature (float): Sampling temperature (0.0 to 2.0).
            max_tokens (int): Maximum tokens to generate.
            **kwargs: Additional provider-specific parameters.
        """
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client: Any = None
        self._initialize_client(**kwargs)

    @abstractmethod
    def _initialize_client(self, **kwargs: Any) -> None:
        """Initialize the API client.

        This method should create and configure the provider's API client.

        Args:
            **kwargs: Provider-specific initialization parameters.

        Raises:
            APIKeyNotFoundError: If required API key is missing.
            ProviderNotAvailableError: If required package is not installed.
        """
        pass

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        """Execute a chat completion request with retry logic.

        This method implements exponential backoff retry logic to handle
        transient API failures. It will retry up to 3 times with increasing
        delays between attempts.

        Args:
            messages (list[dict[str, str]]): List of message dictionaries with 'role' and 'content' keys.

        Returns:
            str: The generated text response.

        Raises:
            ChatCompletionError: If the API call fails after all retries.

        Examples:
            >>> messages = [{"role": "user", "content": "Hello"}]
            >>> response = provider.chat_completion(messages)
        """
        try:
            return self._chat_completion_impl(messages)
        except Exception as e:
            raise ChatCompletionError(self.__class__.__name__, e) from e

    @abstractmethod
    def _chat_completion_impl(self, messages: list[dict[str, str]]) -> str:
        """Internal implementation of chat completion.

        This method should be implemented by concrete providers to perform
        the actual API call without retry logic.

        Args:
            messages (list[dict[str, str]]): List of message dictionaries.

        Returns:
            str: The generated text response.

        Raises:
            Exception: Any API-specific exceptions.
        """
        pass

    def chat_completion_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Execute chat completion with function/tool calling support.

        Args:
            messages (list[dict[str, str]]): List of message dictionaries.
            tools (list[dict]): List of tool/function definitions.
            tool_choice (str | dict | None): Controls which tool is called ("auto", "none", specific tool).

        Returns:
            dict: Dictionary with 'content' (str or None) and 'tool_calls' (list or None).

        Raises:
            ChatCompletionError: If the API call fails.
            NotImplementedError: If provider doesn't support tools.

        Examples:
            >>> tools = [{
            ...     "type": "function",
            ...     "function": {
            ...         "name": "get_weather",
            ...         "description": "Get weather for a location",
            ...         "parameters": {
            ...             "type": "object",
            ...             "properties": {
            ...                 "location": {"type": "string"}
            ...             }
            ...         }
            ...     }
            ... }]
            >>> result = provider.chat_completion_with_tools(messages, tools)
        """
        try:
            return self._chat_completion_with_tools_impl(messages, tools, tool_choice)
        except NotImplementedError as err:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support tool calling"
            ) from err
        except Exception as e:
            raise ChatCompletionError(self.__class__.__name__, e) from e

    def _chat_completion_with_tools_impl(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Internal implementation of tool calling.

        Should be overridden by providers that support tools.

        Args:
            messages (list[dict[str, str]]): List of message dictionaries.
            tools (list[dict]): List of tool definitions.
            tool_choice (str | dict | None): Tool choice parameter.

        Returns:
            dict: Dict with 'content' and 'tool_calls' keys.

        Raises:
            NotImplementedError: If provider doesn't support tools.
        """
        raise NotImplementedError("Tool calling not implemented for this provider")

    def chat_completion_with_files(
        self,
        messages: list[dict[str, str]],
        files: list[str] | None = None,
    ) -> str:
        """Execute chat completion with file uploads.

        This method allows sending files (images, PDFs) along with the chat messages.
        File support varies by provider:
        - OpenAI: Images (PNG, JPEG, WEBP, GIF), PDFs
        - Gemini: Images, PDFs, Videos, Audio
        - Groq: Limited vision support
        - Ollama: Vision models only

        Args:
            messages (list[dict[str, str]]): List of message dictionaries with 'role' and 'content' keys.
            files (list[str] | None): List of file paths to upload. Supported formats depend on provider.

        Returns:
            str: The generated text response.

        Raises:
            FileUploadNotSupportedError: If provider doesn't support file uploads.
            FileNotFoundError: If a specified file doesn't exist.
            ChatCompletionError: If the API call fails.

        Examples:
            >>> messages = [{"role": "user", "content": "Describe this image"}]
            >>> response = provider.chat_completion_with_files(
            ...     messages,
            ...     files=["image.jpg"]
            ... )

            >>> # Multiple files
            >>> response = provider.chat_completion_with_files(
            ...     messages,
            ...     files=["document.pdf", "chart.png"]
            ... )
        """
        try:
            return self._chat_completion_with_files_impl(messages, files)
        except NotImplementedError as err:
            from ..exceptions import FileUploadNotSupportedError

            raise FileUploadNotSupportedError(
                self.__class__.__name__, "Provider does not support file uploads"
            ) from err
        except Exception as e:
            raise ChatCompletionError(self.__class__.__name__, e) from e

    def _chat_completion_with_files_impl(
        self,
        messages: list[dict[str, str]],
        files: list[str] | None = None,
    ) -> str:
        """Internal implementation of file upload chat completion.

        Should be overridden by providers that support file uploads.

        Args:
            messages (list[dict[str, str]]): List of message dictionaries.
            files (list[str] | None): List of file paths.

        Returns:
            str: Generated text response.

        Raises:
            NotImplementedError: If provider doesn't support file uploads.
        """
        raise NotImplementedError("File upload not implemented for this provider")

    def chat_completion_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream response tokens as they arrive.

        This method returns an iterator that yields response tokens as they
        are generated by the LLM, enabling real-time display of responses.

        Args:
            messages (list[dict[str, str]]): List of message dictionaries with 'role' and 'content' keys.

        Yields:
            str: Individual tokens or chunks of the response text.

        Raises:
            StreamingNotSupportedError: If streaming is not supported.
            ChatCompletionError: If the streaming API call fails.

        Examples:
            >>> messages = [{"role": "user", "content": "Tell me a story"}]
            >>> for chunk in provider.chat_completion_stream(messages):
            ...     print(chunk, end="", flush=True)
        """
        try:
            return self._chat_completion_stream_impl(messages)
        except NotImplementedError as err:
            from ..exceptions import StreamingNotSupportedError

            raise StreamingNotSupportedError(
                self.__class__.__name__, "Provider does not implement streaming"
            ) from err
        except Exception as e:
            raise ChatCompletionError(self.__class__.__name__, e) from e

    def _chat_completion_stream_impl(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Internal implementation of streaming chat completion.

        This method can be overridden by concrete providers that support
        streaming. Default implementation raises NotImplementedError.

        Args:
            messages (list[dict[str, str]]): List of message dictionaries.

        Yields:
            str: Individual tokens or chunks of the response text.

        Raises:
            NotImplementedError: If provider doesn't support streaming.
        """
        raise NotImplementedError("Streaming not implemented for this provider")

    @staticmethod
    @abstractmethod
    def get_default_model() -> str:
        """Get the default model name for this provider.

        Returns:
            Default model name as string.
        """
        pass

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Check if the provider's package is installed.

        Returns:
            True if the provider can be used, False otherwise.
        """
        pass

    def __repr__(self) -> str:
        """Return string representation of the provider.

        Returns:
            String with provider info.
        """
        return (
            f"{self.__class__.__name__}(" f"model={self.llm}, " f"temperature={self.temperature})"
        )
