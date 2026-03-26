"""Concrete implementations of LLM providers with streaming support."""

from collections.abc import Iterator
from typing import Any

from ..exceptions import APIKeyNotFoundError, ProviderNotAvailableError
from ..utils.logging_config import get_logger
from .base_provider import BaseProvider

logger = get_logger(__name__)

# Optional imports
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

try:
    from groq import APIStatusError, Groq
except ImportError:
    Groq = None  # type: ignore
    APIStatusError = None  # type: ignore

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
        logger.debug("Initializing OpenAI provider")

        if not self.is_available():
            logger.error("OpenAI package not available")
            raise ProviderNotAvailableError("openai", "openai")

        api_key = kwargs.get("api_key")
        if not api_key:
            logger.error("OpenAI API key not found")
            raise APIKeyNotFoundError("openai", "OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key)
        logger.info(f"OpenAI client initialized with model {self.llm}")

    def _chat_completion_impl(self, messages: list[dict[str, str]]) -> str | None:
        """Execute chat completion with OpenAI.

        Args:
            messages: List of message dictionaries.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If client is not initialized.
        """
        if not self.client:
            logger.error("OpenAI client not initialized")
            raise RuntimeError("OpenAI client not initialized")

        logger.debug(f"Calling OpenAI API: model={self.llm}, messages={len(messages)}")

        response = self.client.chat.completions.create(
            model=self.llm,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content

        if content is None:
            logger.debug("OpenAI response content is None")
            return None

        logger.debug(f"OpenAI response received: {len(content)} characters")
        return content

    def _chat_completion_with_files_impl(
        self,
        messages: list[dict[str, str]],
        files: list[str] | None = None,
    ) -> str:
        """Execute chat completion with files using OpenAI.

        Args:
            messages: List of message dictionaries.
            files: List of file paths to upload.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If client is not initialized.
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        from ..utils.file_utils import prepare_files_for_provider

        logger.debug(f"Calling OpenAI API with {len(files or [])} files")

        # Prepare messages with files
        enhanced_messages = messages.copy()

        if files:
            # Prepare file data
            file_data = prepare_files_for_provider(files, "openai")

            # Add files to the last user message or create new message
            if enhanced_messages and enhanced_messages[-1]["role"] == "user":
                # Convert content to list format if it's a string
                last_msg = enhanced_messages[-1]
                if isinstance(last_msg["content"], str):
                    text_content = last_msg["content"]
                    last_msg["content"] = [{"type": "text", "text": text_content}]

                # Add files
                last_msg["content"].extend(file_data)
            else:
                # Create new user message with files
                enhanced_messages.append({"role": "user", "content": file_data})

        response = self.client.chat.completions.create(
            model=self.llm,
            messages=enhanced_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content
        if content:
            logger.debug(f"OpenAI response with files: {len(content)} characters")
        return content

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
            logger.error("OpenAI client not initialized")
            raise RuntimeError("OpenAI client not initialized")

        logger.debug(f"Starting OpenAI streaming: model={self.llm}")

        stream = self.client.chat.completions.create(
            model=self.llm,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        chunk_count = 0
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                chunk_count += 1
                yield chunk.choices[0].delta.content

        logger.debug(f"OpenAI streaming completed: {chunk_count} chunks")

    def _chat_completion_with_tools_impl(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Execute chat completion with tools."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        logger.debug(f"Calling OpenAI with {len(tools)} tools")

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

        result = {
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

        if result["tool_calls"]:
            logger.debug(f"Tools called: {[tc['function']['name'] for tc in result['tool_calls']]}")

        return result

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
        available = OpenAI is not None
        if not available:
            logger.debug("OpenAI package not available")
        return available


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
        logger.debug("Initializing Groq provider")

        if not self.is_available():
            logger.error("Groq package not available")
            raise ProviderNotAvailableError("groq", "groq")

        api_key = kwargs.get("api_key")
        if not api_key:
            logger.error("Groq API key not found")
            raise APIKeyNotFoundError("groq", "GROQ_API_KEY")

        self.client = Groq(api_key=api_key)
        logger.info(f"Groq client initialized with model {self.llm}")

    def _chat_completion_impl(self, messages: list[dict[str, str]]) -> str | None:
        """Execute chat completion with Groq.

        Args:
            messages: List of message dictionaries.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If client is not initialized.
        """
        if not self.client:
            logger.error("Groq client not initialized")
            raise RuntimeError("Groq client not initialized")

        logger.debug(f"Calling Groq API: model={self.llm}, messages={len(messages)}")

        try:
            response = self.client.chat.completions.create(
                model=self.llm,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            error_str = str(e)
            logger.debug(f"Groq API error caught: {error_str}")
            if ((hasattr(e, "status_code") and e.status_code == 413) or "413" in error_str) and (
                "rate_limit_exceeded" in error_str or "Rate limit exceeded" in error_str
            ):
                logger.warning(
                    f"Groq TPM limit exceeded for model {self.llm}. Attempting fallback."
                )
                fallback_model = self._find_fallback_model(error_str)
                if fallback_model:
                    logger.info(f"Retrying with fallback model: {fallback_model}")
                    self.llm = fallback_model
                    return self._chat_completion_impl(messages)
            raise

        content = response.choices[0].message.content

        if content is None:
            logger.debug("Groq response content is None")
            return None

        logger.debug(f"Groq response received: {len(content)} characters")
        return content

    def _find_fallback_model(self, error_message: str) -> str | None:
        """Find a model with higher TPM limit from the saved rate limits.

        Args:
            error_message: The API error message containing requested tokens.

        Returns:
            Name of a suitable fallback model or None.
        """
        import re
        from pathlib import Path

        # Extract requested tokens from message e.g. "Requested 21142"
        match = re.search(r"Requested (\d+)", error_message)
        if not match:
            logger.debug("Could not parse requested tokens from error message")
            return None

        requested_tokens = int(match.group(1))
        logger.debug(f"Parsed requested tokens: {requested_tokens}")

        # Load rate limits from Markdown file
        limits_file = Path(__file__).parent / "groq_rate_limits.md"
        if not limits_file.exists():
            logger.warning(f"Rate limits file not found at {limits_file}")
            return None

        try:
            with open(limits_file) as f:
                lines = f.readlines()

            # Skip header lines
            data_lines = [
                line
                for line in lines
                if "|" in line and "---" not in line and "MODEL ID" not in line
            ]

            best_model = None
            highest_tpm = 0

            for line in data_lines:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 4:
                    model_id = parts[0]
                    # Skip "groq/compound" models as requested (they are specialized)
                    if "groq/compound" in model_id:
                        continue

                    tpm_str = parts[3]

                    # Convert TPM string (e.g. "30K", "1.2K", "6000") to integer
                    tpm = 0
                    if tpm_str != "-":
                        if "K" in tpm_str:
                            tpm = int(float(tpm_str.replace("K", "")) * 1000)
                        else:
                            tpm = int(tpm_str.replace(",", ""))

                    if tpm > requested_tokens and (best_model is None or tpm > highest_tpm):
                        # We found a model that can handle the request.
                        # Prefer models with higher TPM.
                        highest_tpm = tpm
                        best_model = model_id

            return best_model
        except Exception as e:
            logger.error(f"Error parsing rate limits file: {e}")
            return None

    def _chat_completion_with_tools_impl(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Execute chat completion with tools."""
        if not self.client:
            raise RuntimeError("Groq client not initialized")

        logger.debug(f"Calling Groq with {len(tools)} tools")

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

    def _chat_completion_with_files_impl(
        self,
        messages: list[dict[str, str]],
        files: list[str] | None = None,
    ) -> str:
        """Execute chat completion with files using Groq.

        Note: Groq has limited vision support. Only certain models support images.

        Args:
            messages: List of message dictionaries.
            files: List of file paths (images only).

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If client is not initialized.
            ValueError: If non-image files are provided.
        """
        if not self.client:
            raise RuntimeError("Groq client not initialized")

        from ..utils.file_utils import detect_file_type, prepare_files_for_provider

        logger.debug(f"Calling Groq API with {len(files or [])} files")

        # Validate files (Groq only supports images for vision models)
        if files:
            for file_path in files:
                if detect_file_type(file_path) != "image":
                    raise ValueError("Groq only supports image files for vision models")

        # Prepare messages with files
        enhanced_messages = messages.copy()

        if files:
            file_data = prepare_files_for_provider(files, "groq")

            if enhanced_messages and enhanced_messages[-1]["role"] == "user":
                last_msg = enhanced_messages[-1]
                if isinstance(last_msg["content"], str):
                    text_content = last_msg["content"]
                    last_msg["content"] = [{"type": "text", "text": text_content}]

                last_msg["content"].extend(file_data)
            else:
                enhanced_messages.append({"role": "user", "content": file_data})

        response = self.client.chat.completions.create(
            model=self.llm,
            messages=enhanced_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content
        if content:
            logger.debug(f"Groq response with files: {len(content)} characters")
        return content

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

        logger.debug(f"Starting Groq streaming: model={self.llm}")

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
        available = Groq is not None
        if not available:
            logger.debug("Groq package not available")
        return available


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
        logger.debug("Initializing Gemini provider")

        if not self.is_available():
            logger.error("OpenAI package not available (required for Gemini)")
            raise ProviderNotAvailableError("gemini", "openai")

        api_key = kwargs.get("api_key")
        if not api_key:
            logger.error("Gemini API key not found")
            raise APIKeyNotFoundError("gemini", "GEMINI_API_KEY")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        logger.info(f"Gemini client initialized with model {self.llm}")

    def _chat_completion_impl(self, messages: list[dict[str, str]]) -> str | None:
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

        logger.debug(f"Calling Gemini API: model={self.llm}")

        response = self.client.chat.completions.create(
            model=self.llm,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content

        if content is None:
            logger.debug("Gemini response content is None")
            return None

        logger.debug(f"Gemini response received: {len(content)} characters")
        return content

    def _chat_completion_with_tools_impl(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Execute chat completion with tools."""
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        logger.debug(f"Calling Gemini with {len(tools)} tools")

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

    def _chat_completion_with_files_impl(
        self,
        messages: list[dict[str, str]],
        files: list[str] | None = None,
    ) -> str:
        """Execute chat completion with files using Gemini.

        Args:
            messages: List of message dictionaries.
            files: List of file paths to upload.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If client is not initialized.
        """
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        from ..utils.file_utils import prepare_files_for_provider

        logger.debug(f"Calling Gemini API with {len(files or [])} files")

        # Prepare messages with files
        enhanced_messages = messages.copy()

        if files:
            # Prepare file data
            file_data = prepare_files_for_provider(files, "gemini")

            # Add files to the last user message
            if enhanced_messages and enhanced_messages[-1]["role"] == "user":
                last_msg = enhanced_messages[-1]
                if isinstance(last_msg["content"], str):
                    text_content = last_msg["content"]
                    last_msg["content"] = [{"type": "text", "text": text_content}]

                last_msg["content"].extend(file_data)
            else:
                enhanced_messages.append({"role": "user", "content": file_data})

        response = self.client.chat.completions.create(
            model=self.llm,
            messages=enhanced_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content
        if content:
            logger.debug(f"Gemini response with files: {len(content)} characters")
        return content

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

        logger.debug(f"Starting Gemini streaming: model={self.llm}")

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
        available = OpenAI is not None
        if not available:
            logger.debug("OpenAI package not available (required for Gemini)")
        return available


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

        if llm.endswith("-cloud") and not use_cloud:
            self.use_cloud = True
            logger.debug(f"Auto-detected cloud mode from model name: {llm}")

        super().__init__(llm, temperature, max_tokens, **kwargs)

    def _initialize_client(self, **kwargs: Any) -> None:
        """Initialize Ollama client (local or cloud).

        Args:
            **kwargs: May contain 'api_key' for Ollama Cloud.

        Raises:
            ProviderNotAvailableError: If ollama package is not installed.
            APIKeyNotFoundError: If cloud mode but API key is missing.
        """
        mode = "cloud" if self.use_cloud else "local"
        logger.debug(f"Initializing Ollama provider in {mode} mode")

        if not self.is_available():
            logger.error("Ollama package not available")
            raise ProviderNotAvailableError("ollama", "ollama")

        if self.use_cloud:
            # Ollama Cloud mode
            api_key = kwargs.get("api_key") or self._api_key
            if not api_key:
                logger.error("Ollama Cloud API key not found")
                raise APIKeyNotFoundError("ollama_cloud", "OLLAMA_API_KEY")

            # Create client with cloud settings
            self.client = Client(
                host=self.host or "https://ollama.com",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            logger.info(f"Ollama Cloud client initialized with model {self.llm}")
        else:
            # Local Ollama mode
            if self.host:
                self.client = Client(host=self.host)
                logger.info(f"Ollama local client initialized with custom host: {self.host}")
            else:
                # Use default local client
                self.client = Client()
                logger.info(f"Ollama local client initialized with model {self.llm}")

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

        mode = "cloud" if self.use_cloud else "local"
        logger.debug(f"Calling Ollama ({mode}): model={self.llm}, messages={len(messages)}")

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
        content = response["message"]["content"]
        logger.debug(f"Ollama response received: {len(content)} characters")
        return content

    def _chat_completion_with_tools_impl(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Tool calling support for Ollama."""
        if not self.is_available():
            raise ProviderNotAvailableError("ollama", "ollama")

        logger.warning("Tool calling in Ollama is experimental")
        raise NotImplementedError(
            "Tool calling support in Ollama is experimental. "
            "Please check Ollama documentation for current status."
        )

    def _chat_completion_with_files_impl(
        self,
        messages: list[dict[str, str]],
        files: list[str] | None = None,
    ) -> str:
        """Execute chat completion with files using Ollama.

        Note: Requires a vision-capable model like llava or bakllava.

        Args:
            messages: List of message dictionaries.
            files: List of file paths (images only).

        Returns:
            Generated text response.

        Raises:
            ProviderNotAvailableError: If ollama package is not available.
            ValueError: If non-image files are provided.
        """
        if not self.is_available():
            raise ProviderNotAvailableError("ollama", "ollama")

        from ..utils.file_utils import detect_file_type, encode_file_base64

        logger.debug(f"Calling Ollama with {len(files or [])} files")

        # Validate files (Ollama vision models only support images)
        if files:
            for file_path in files:
                if detect_file_type(file_path) != "image":
                    raise ValueError("Ollama vision models only support image files")

        # Prepare messages with files
        enhanced_messages = messages.copy()

        if files:
            # Ollama uses base64 encoded images in a different format
            images = [encode_file_base64(f) for f in files]

            # Find last user message
            if enhanced_messages and enhanced_messages[-1]["role"] == "user":
                last_msg = enhanced_messages[-1]
                # Ollama expects images as a separate field
                last_msg["images"] = images
            else:
                enhanced_messages.append({"role": "user", "content": "", "images": images})

        options = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }

        if not self.use_cloud:
            options.update({"repeat_penalty": 1.2, "top_k": 10, "top_p": 0.5})

        kwargs = {
            "model": self.llm,
            "messages": enhanced_messages,
            "stream": False,
            "options": options,
        }

        if not self.use_cloud:
            kwargs["keep_alive"] = self.keep_alive

        response = self.client.chat(**kwargs)
        content = response["message"]["content"]
        logger.debug(f"Ollama response with files: {len(content)} characters")
        return content

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

        mode = "cloud" if self.use_cloud else "local"
        logger.debug(f"Starting Ollama ({mode}) streaming: model={self.llm}")

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
        available = OLLAMA_AVAILABLE
        if not available:
            logger.debug("Ollama package not available")
        return available

    def __repr__(self) -> str:
        """Return string representation of the provider.

        Returns:
            String with provider info.
        """
        mode = "cloud" if self.use_cloud else "local"
        return (
            f"OllamaProvider(model={self.llm}, " f"temperature={self.temperature}, " f"mode={mode})"
        )
