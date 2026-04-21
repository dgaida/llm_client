"""LLM Client with token counting, async support, and config file loading."""

import os
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from .config import LLMConfig
from .providers.base_provider import BaseProvider
from .providers.provider_factory import ProviderFactory
from .utils.logging_config import get_logger
from .utils.token_counter import TokenCounter

logger = get_logger(__name__)


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
        use_ollama_cloud: bool = False,
        ollama_host: str | None = None,
    ) -> None:
        """Initialize the LLM Client.

        Args:
            llm (str | None): Model name. If None, uses provider's default.
            temperature (float): Sampling temperature (0.0 to 2.0).
            max_tokens (int): Maximum tokens to generate.
            api_choice (str | None): Explicit API choice. If None, auto-selects.
            secrets_path (str): Path to secrets.env file.
            keep_alive (str): Ollama-specific keep-alive duration.
            use_async (bool): If True, use async providers.
            use_ollama_cloud (bool): If True, use Ollama Cloud API.
            ollama_host (str | None): Custom Ollama host URL.

        Examples:
            >>> client = LLMClient(llm="gpt-4o", temperature=0.5)
            >>> client = LLMClient(api_choice="gemini")
            >>> async_client = LLMClient(use_async=True)
        """
        logger.debug(f"Initializing LLMClient with api_choice={api_choice}, llm={llm}")

        # Load environment variables
        if os.path.exists(secrets_path):
            logger.debug(f"Loading secrets from {secrets_path}")
            load_dotenv(secrets_path)

        # Load API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY")
        self.api_key = os.getenv("API_KEY")

        # Log which keys are available (without exposing values)
        available_keys = []
        if self.openai_api_key:
            available_keys.append("OpenAI")
        if self.groq_api_key:
            available_keys.append("Groq")
        if self.gemini_api_key:
            available_keys.append("Gemini")
        if self.ollama_api_key:
            available_keys.append("Ollama")
        if self.api_key:
            available_keys.append("Generic (API_KEY)")

        if available_keys:
            logger.debug(f"Found API keys for: {', '.join(available_keys)}")
        else:
            logger.debug("No API keys found in environment, will use Ollama")

        # Store Ollama-specific settings
        self.use_ollama_cloud = use_ollama_cloud
        self.ollama_host = ollama_host

        # Try loading from Google Colab userdata
        self._load_colab_secrets(api_choice)

        # Store configuration
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive
        self._user_specified_llm = llm
        self.use_async = use_async

        # Initialize token counter
        self.token_counter = TokenCounter()
        logger.debug("Initialized TokenCounter")

        # Create provider using factory
        logger.info(f"Creating provider for API: {api_choice or 'auto-detect'}")
        self.provider: BaseProvider = ProviderFactory.create_provider(
            api_choice=api_choice,
            llm=llm,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=self.openai_api_key,
            groq_api_key=self.groq_api_key,
            gemini_api_key=self.gemini_api_key,
            ollama_api_key=self.ollama_api_key,
            api_key=self.api_key,
            keep_alive=keep_alive,
            use_async=use_async,
            use_ollama_cloud=use_ollama_cloud,
            ollama_host=ollama_host,
        )

        # Store current API choice
        self.api_choice = self._get_api_choice_from_provider()
        logger.info(f"Initialized with provider: {self.api_choice}, model: {self.llm}")

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
            config_path (str | Path): Path to YAML or JSON configuration file.
            provider (str | None): Provider to use. If None, uses default from config.
            secrets_path (str): Path to secrets.env file.
            use_async (bool): If True, use async providers.

        Returns:
            LLMClient: Configured LLMClient instance.

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
        logger.info(f"Loading LLMClient from config: {config_path}")

        # Load configuration
        config = LLMConfig.from_file(config_path)

        # Validate configuration
        is_valid, errors = config.validate()
        if not is_valid:
            error_msg = f"Invalid configuration: {'; '.join(errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug("Configuration validated successfully")

        # Determine which provider to use
        provider_name = provider or config.default_provider
        provider_config = config.get_provider_config(provider_name)
        logger.debug(f"Using provider: {provider_name}")

        # Extract parameters
        llm = provider_config.get("model")
        temperature = provider_config.get("temperature", 0.7)
        max_tokens = provider_config.get("max_tokens", 512)
        keep_alive = provider_config.get("keep_alive", "5m")
        use_ollama_cloud = provider_config.get("use_cloud", False)
        ollama_host = provider_config.get("host")

        # Create client
        return cls(
            llm=llm,
            temperature=temperature,
            max_tokens=max_tokens,
            api_choice=provider_name,
            secrets_path=secrets_path,
            keep_alive=keep_alive,
            use_async=use_async,
            use_ollama_cloud=use_ollama_cloud,
            ollama_host=ollama_host,
        )

    def _load_colab_secrets(self, api_choice: str | None = None) -> None:
        """Load API keys from Google Colab userdata if available.

        Args:
            api_choice (str | None): If specified, only load the key for this provider.
                       If None, load all available keys for auto-selection.
        """
        if "google.colab" not in sys.modules and "COLAB_GPU" not in os.environ:
            return

        logger.debug("Detected Google Colab environment, attempting to load secrets")

        try:
            from google.colab import userdata

            # Try loading generic API_KEY if not already set
            if not self.api_key:
                try:
                    self.api_key = userdata.get("API_KEY")
                    logger.debug("Loaded API_KEY from Colab userdata")
                except Exception:
                    pass

            if api_choice:
                api_choice_lower = api_choice.lower()
                key_map = {
                    "openai": ("OPENAI_API_KEY", "openai_api_key"),
                    "groq": ("GROQ_API_KEY", "groq_api_key"),
                    "gemini": ("GEMINI_API_KEY", "gemini_api_key"),
                    "ollama": ("OLLAMA_API_KEY", "ollama_api_key"),
                }

                if api_choice_lower in key_map:
                    env_key, attr_name = key_map[api_choice_lower]
                    current_value = getattr(self, attr_name)
                    if not current_value:
                        try:
                            setattr(self, attr_name, userdata.get(env_key))
                            logger.debug(f"Loaded {env_key} from Colab userdata")
                        except Exception as e:
                            logger.debug(f"{env_key} not found in Colab secrets: {e}")
            else:
                # Load all keys for auto-selection
                for key_name, attr_name in [
                    ("OPENAI_API_KEY", "openai_api_key"),
                    ("GROQ_API_KEY", "groq_api_key"),
                    ("GEMINI_API_KEY", "gemini_api_key"),
                    ("OLLAMA_API_KEY", "ollama_api_key"),
                ]:
                    if not getattr(self, attr_name):
                        try:
                            setattr(self, attr_name, userdata.get(key_name))
                            logger.debug(f"Loaded {key_name} from Colab userdata")
                        except Exception as e:
                            logger.debug(f"Could not load {key_name}: {e}")
        except Exception as e:
            logger.debug(f"Could not access Colab userdata: {e}")

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
        use_ollama_cloud: bool | None = None,
    ) -> None:
        """Switch to a different LLM provider at runtime.

        This allows changing providers without creating a new client instance.
        Useful for fallback strategies, cost optimization, or A/B testing.

        Args:
            api_choice (str): Target API to switch to.
            llm (str | None): Optional new model name.
            temperature (float | None): Optional new temperature.
            max_tokens (int | None): Optional new max_tokens.
            use_ollama_cloud (bool | None): Optional cloud mode setting.

        Raises:
            InvalidProviderError: If api_choice is invalid.
            APIKeyNotFoundError: If API key for chosen provider is missing.
            ProviderNotAvailableError: If provider package is not installed.

        Examples:
            >>> client = LLMClient(api_choice="openai")
            >>> client.switch_provider("gemini", llm="gemini-2.5-flash")
            >>> client.switch_provider("groq", temperature=0.3)
        """
        logger.info(f"Switching provider from {self.api_choice} to {api_choice}")

        # Update parameters if provided
        if temperature is not None:
            self.temperature = temperature
            logger.debug(f"Updated temperature to {temperature}")
        if max_tokens is not None:
            self.max_tokens = max_tokens
            logger.debug(f"Updated max_tokens to {max_tokens}")
        if use_ollama_cloud is not None:
            self.use_ollama_cloud = use_ollama_cloud
            logger.debug(f"Updated use_ollama_cloud to {use_ollama_cloud}")

        # Update user-specified model
        self._user_specified_llm = llm

        # Load missing API keys from Colab
        self._load_colab_secrets(api_choice)

        # Create new provider
        self.provider = ProviderFactory.create_provider(
            api_choice=api_choice,
            llm=llm,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=self.openai_api_key,
            groq_api_key=self.groq_api_key,
            gemini_api_key=self.gemini_api_key,
            ollama_api_key=self.ollama_api_key,
            keep_alive=self.keep_alive,
            use_async=self.use_async,
            use_ollama_cloud=self.use_ollama_cloud,
            ollama_host=self.ollama_host,
        )

        # Update API choice
        self.api_choice = api_choice.lower()
        logger.info(f"Successfully switched to {self.api_choice} with model {self.llm}")

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        """Execute a chat completion using the current provider.

        This method includes automatic retry logic with exponential backoff
        to handle transient API failures.

        Args:
            messages (list[dict[str, str]]): List of message dicts with 'role' and 'content' keys.

        Returns:
            str: Generated text response.

        Raises:
            ChatCompletionError: If the provider call fails after retries.

        Examples:
            >>> messages = [{"role": "user", "content": "Hello"}]
            >>> response = client.chat_completion(messages)
        """
        logger.debug(f"Executing chat completion with {len(messages)} messages")
        response = self.provider.chat_completion(messages)

        if response is not None:
            logger.debug(f"Chat completion successful, response length: {len(response)}")
        return response

    def chat_completion_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Execute chat completion with function/tool calling.

        This method enables the LLM to call external functions/tools during
        generation. The LLM can decide which tools to call based on the context.

        Args:
            messages (list[dict[str, str]]): List of message dicts with 'role' and 'content' keys.
            tools (list[dict]): List of tool definitions in OpenAI format.
            tool_choice (str | dict | None): Controls tool selection:
                - "auto" (default): LLM decides whether to call tools
                - "none": LLM will not call any tools
                - {"type": "function", "function": {"name": "..."}}: Force specific tool

        Returns:
            dict: Dictionary containing:
                - 'content': Generated text (str or None if tool called)
                - 'tool_calls': List of tool calls (or None if no tools called)
                    Each tool call has: id, type, function (with name and arguments)

        Raises:
            NotImplementedError: If provider doesn't support tool calling.
            ChatCompletionError: If the API call fails.

        Examples:
            >>> tools = [{
            ...     "type": "function",
            ...     "function": {
            ...         "name": "get_current_weather",
            ...         "description": "Get the current weather in a location",
            ...         "parameters": {
            ...             "type": "object",
            ...             "properties": {
            ...                 "location": {
            ...                     "type": "string",
            ...                     "description": "City and state, e.g. San Francisco, CA"
            ...                 },
            ...                 "unit": {
            ...                     "type": "string",
            ...                     "enum": ["celsius", "fahrenheit"]
            ...                 }
            ...             },
            ...             "required": ["location"]
            ...         }
            ...     }
            ... }]
            >>> messages = [{"role": "user", "content": "What's the weather in Boston?"}]
            >>> result = client.chat_completion_with_tools(messages, tools)
            >>> if result['tool_calls']:
            ...     for tool_call in result['tool_calls']:
            ...         print(f"Calling: {tool_call['function']['name']}")
            ...         print(f"Arguments: {tool_call['function']['arguments']}")
        """
        logger.debug(f"Executing chat completion with {len(tools)} tools")
        result = self.provider.chat_completion_with_tools(messages, tools, tool_choice)
        if result.get("tool_calls"):
            logger.debug(f"Tools called: {[tc['function']['name'] for tc in result['tool_calls']]}")
        return result

    def chat_completion_with_files(
        self,
        messages: list[dict[str, str]],
        files: list[str] | None = None,
    ) -> str:
        """Execute chat completion with file uploads.

        This method allows uploading files (images, PDFs, etc.) along with chat messages.
        File support varies by provider:
        - OpenAI: Images (PNG, JPEG, WEBP, GIF), PDFs (GPT-4o and newer)
        - Gemini: Images, PDFs, Videos, Audio files
        - Groq: Images only (vision models)
        - Ollama: Images only (requires vision models like llava)

        Args:
            messages (list[dict[str, str]]): List of message dicts with 'role' and 'content' keys.
            files (list[str] | None): List of file paths to upload. If None, works like regular chat_completion.

        Returns:
            str: Generated text response.

        Raises:
            FileUploadNotSupportedError: If provider doesn't support file uploads.
            FileNotFoundError: If a specified file doesn't exist.
            ValueError: If file type is not supported by the provider.
            ChatCompletionError: If the API call fails.

        Examples:
            >>> # Single image
            >>> messages = [{"role": "user", "content": "What's in this image?"}]
            >>> response = client.chat_completion_with_files(
            ...     messages,
            ...     files=["vacation_photo.jpg"]
            ... )

            >>> # Multiple files
            >>> messages = [{"role": "user", "content": "Analyze these documents"}]
            >>> response = client.chat_completion_with_files(
            ...     messages,
            ...     files=["report.pdf", "chart.png", "data.jpg"]
            ... )

            >>> # Image analysis with Gemini
            >>> client = LLMClient(api_choice="gemini")
            >>> messages = [{"role": "user", "content": "Describe this image in detail"}]
            >>> response = client.chat_completion_with_files(
            ...     messages,
            ...     files=["complex_diagram.png"]
            ... )

            >>> # PDF analysis with OpenAI
            >>> client = LLMClient(api_choice="openai", llm="gpt-4o")
            >>> messages = [{"role": "user", "content": "Summarize this document"}]
            >>> response = client.chat_completion_with_files(
            ...     messages,
            ...     files=["research_paper.pdf"]
            ... )
        """
        logger.debug(
            f"Executing chat completion with files: {len(messages)} messages, "
            f"{len(files) if files else 0} files"
        )

        if files:
            # Validate files exist
            from pathlib import Path

            for file_path in files:
                if not Path(file_path).exists():
                    raise FileNotFoundError(f"File not found: {file_path}")

            logger.debug(f"Files to upload: {files}")

        response = self.provider.chat_completion_with_files(messages, files)

        if response is not None:
            logger.debug(f"Chat completion with files successful, response length: {len(response)}")
        return response

    def chat_completion_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream response tokens as they arrive from the LLM.

        This method returns an iterator that yields response tokens in real-time,
        enabling progressive display of the response.

        Args:
            messages (list[dict[str, str]]): List of message dicts with 'role' and 'content' keys.

        Yields:
            str: Individual tokens or chunks of the response text.

        Raises:
            StreamingNotSupportedError: If streaming is not supported.
            ChatCompletionError: If the streaming API call fails.

        Examples:
            >>> messages = [{"role": "user", "content": "Tell me a story"}]
            >>> for chunk in client.chat_completion_stream(messages):
            ...     print(chunk, end="", flush=True)
            >>> print()  # New line after streaming completes
        """
        logger.debug(f"Starting streaming chat completion with {len(messages)} messages")
        return self.provider.chat_completion_stream(messages)

    async def achat_completion(self, messages: list[dict[str, str]]) -> str:
        """Execute async chat completion.

        Args:
            messages (list[dict[str, str]]): List of message dicts.

        Returns:
            str: Generated text response.

        Raises:
            RuntimeError: If provider doesn't support async.

        Examples:
            >>> response = await client.achat_completion(messages)
        """
        logger.debug(f"Executing async chat completion with {len(messages)} messages")
        if not hasattr(self.provider, "achat_completion"):
            error_msg = (
                f"{self.provider.__class__.__name__} does not support async. "
                f"Create client with use_async=True"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        response = await self.provider.achat_completion(messages)
        logger.debug("Async chat completion successful")
        return response

    async def achat_completion_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Execute async chat completion with tools.

        Args:
            messages (list[dict[str, str]]): List of message dicts.
            tools (list[dict]): List of tool definitions.
            tool_choice (str | dict | None): Tool selection control.

        Returns:
            dict: Dict with 'content' and 'tool_calls' keys.

        Raises:
            RuntimeError: If provider doesn't support async tools.

        Examples:
            >>> result = await client.achat_completion_with_tools(messages, tools)
        """
        logger.debug(f"Executing async chat completion with {len(tools)} tools")
        if not hasattr(self.provider, "achat_completion_with_tools"):
            error_msg = (
                f"{self.provider.__class__.__name__} does not support async tools. "
                f"Create client with use_async=True"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        return await self.provider.achat_completion_with_tools(messages, tools, tool_choice)

    async def achat_completion_with_files(
        self,
        messages: list[dict[str, str]],
        files: list[str] | None = None,
    ) -> str:
        """Execute async chat completion with file uploads.

        Args:
            messages (list[dict[str, str]]): List of message dicts.
            files (list[str] | None): List of file paths to upload.

        Returns:
            str: Generated text response.

        Raises:
            RuntimeError: If provider doesn't support async file uploads.
            FileUploadNotSupportedError: If provider doesn't support file uploads.

        Examples:
            >>> response = await client.achat_completion_with_files(
            ...     messages,
            ...     files=["image.jpg"]
            ... )
        """
        logger.debug(
            f"Executing async chat completion with files: {len(messages)} messages, "
            f"{len(files) if files else 0} files"
        )

        if not hasattr(self.provider, "achat_completion_with_files"):
            error_msg = (
                f"{self.provider.__class__.__name__} does not support async file uploads. "
                f"Create client with use_async=True"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if files:
            from pathlib import Path

            for file_path in files:
                if not Path(file_path).exists():
                    raise FileNotFoundError(f"File not found: {file_path}")

        response = await self.provider.achat_completion_with_files(messages, files)
        logger.debug("Async chat completion with files successful")
        return response

    async def achat_completion_stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Stream response tokens asynchronously.

        Args:
            messages (list[dict[str, str]]): List of message dicts.

        Yields:
            str: Individual tokens or chunks.

        Raises:
            RuntimeError: If provider doesn't support async streaming.

        Examples:
            >>> async for chunk in client.achat_completion_stream(messages):
            ...     print(chunk, end="", flush=True)
        """
        logger.debug(f"Starting async streaming with {len(messages)} messages")
        if not hasattr(self.provider, "achat_completion_stream"):
            error_msg = f"{self.provider.__class__.__name__} does not support async streaming"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        async for chunk in self.provider.achat_completion_stream(messages):
            yield chunk

    def count_tokens(self, messages: list[dict[str, str]], model: str | None = None) -> int:
        """Count tokens in messages using tiktoken.

        Args:
            messages (list[dict[str, str]]): List of message dicts to count tokens for.
            model (str | None): Model name for encoding. If None, uses current model.

        Returns:
            int: Total token count.

        Examples:
            >>> messages = [{"role": "user", "content": "Hello world"}]
            >>> token_count = client.count_tokens(messages)
            >>> print(f"Tokens: {token_count}")
        """
        model_name = model or self.llm
        token_count = self.token_counter.count_tokens(messages, model=model_name)
        logger.debug(f"Counted {token_count} tokens for {len(messages)} messages")
        return token_count

    def count_string_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens in a string.

        Args:
            text (str): Text to count tokens for.
            model (str | None): Model name. If None, uses current model.

        Returns:
            int: Token count.

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
    def client(self) -> Any:
        """Get the underlying API client (for backward compatibility).

        Returns:
            Any: The provider's client instance.
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
