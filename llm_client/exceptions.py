"""Custom exception classes for llm_client package."""


class LLMClientError(Exception):
    """Base exception for LLM Client.

    All custom exceptions in the llm_client package inherit from this class.
    This allows users to catch all llm_client-specific errors with a single
    except clause.

    Examples:
        >>> try:
        ...     client = LLMClient(api_choice="openai")
        ... except LLMClientError as e:
        ...     print(f"LLM Client error: {e}")
    """

    pass


class APIKeyNotFoundError(LLMClientError):
    """Raised when required API key is missing.

    This exception is raised when attempting to initialize a provider that
    requires an API key, but the key is not found in environment variables
    or passed explicitly.

    Attributes:
        provider: Name of the provider that requires the key.
        key_name: Name of the environment variable for the API key.

    Examples:
        >>> raise APIKeyNotFoundError("openai", "OPENAI_API_KEY")
        APIKeyNotFoundError: OPENAI_API_KEY not found for openai provider.
        Please set it in environment or pass explicitly.
    """

    def __init__(self, provider: str, key_name: str):
        """Initialize the exception.

        Args:
            provider: Name of the provider (e.g., 'openai', 'groq').
            key_name: Name of the required environment variable.
        """
        self.provider = provider
        self.key_name = key_name
        message = (
            f"{key_name} not found for {provider} provider. "
            f"Please set it in environment or pass explicitly."
        )
        super().__init__(message)


class ProviderNotAvailableError(LLMClientError):
    """Raised when requested provider is not available.

    This exception is raised when attempting to use a provider whose
    required package is not installed.

    Attributes:
        provider: Name of the provider that is not available.
        package_name: Name of the required package.

    Examples:
        >>> raise ProviderNotAvailableError("groq", "groq")
        ProviderNotAvailableError: groq provider not available.
        Install with: pip install groq
    """

    def __init__(self, provider: str, package_name: str):
        """Initialize the exception.

        Args:
            provider: Name of the provider (e.g., 'openai', 'groq').
            package_name: Name of the required pip package.
        """
        self.provider = provider
        self.package_name = package_name
        message = f"{provider} provider not available. " f"Install with: pip install {package_name}"
        super().__init__(message)


class InvalidProviderError(LLMClientError):
    """Raised when an invalid provider name is specified.

    This exception is raised when attempting to create or switch to a
    provider that doesn't exist.

    Attributes:
        provider: The invalid provider name that was specified.
        valid_providers: List of valid provider names.

    Examples:
        >>> raise InvalidProviderError("invalid", ["openai", "groq"])
        InvalidProviderError: Invalid provider: invalid.
        Valid providers are: openai, groq
    """

    def __init__(self, provider: str, valid_providers: list[str]):
        """Initialize the exception.

        Args:
            provider: The invalid provider name.
            valid_providers: List of valid provider names.
        """
        self.provider = provider
        self.valid_providers = valid_providers
        message = (
            f"Invalid provider: {provider}. " f"Valid providers are: {', '.join(valid_providers)}"
        )
        super().__init__(message)


class ChatCompletionError(LLMClientError):
    """Raised when chat completion request fails.

    This exception is raised when an API call to the LLM provider fails
    after all retry attempts have been exhausted.

    Attributes:
        provider: Name of the provider that failed.
        original_error: The original exception that caused the failure.

    Examples:
        >>> raise ChatCompletionError("openai", ConnectionError("Network error"))
        ChatCompletionError: Chat completion failed for openai provider:
        Network error
    """

    def __init__(self, provider: str, original_error: Exception):
        """Initialize the exception.

        Args:
            provider: Name of the provider where the error occurred.
            original_error: The original exception that was raised.
        """
        self.provider = provider
        self.original_error = original_error
        message = (
            f"Chat completion failed for {provider} provider: "
            f"{type(original_error).__name__}: {original_error}"
        )
        super().__init__(message)


class StreamingNotSupportedError(LLMClientError):
    """Raised when streaming is requested but not supported.

    This exception is raised when attempting to use streaming with a
    provider or configuration that doesn't support it.

    Attributes:
        provider: Name of the provider.
        reason: Optional reason why streaming is not supported.

    Examples:
        >>> raise StreamingNotSupportedError("custom_provider")
        StreamingNotSupportedError: Streaming not supported for custom_provider
    """

    def __init__(self, provider: str, reason: str | None = None):
        """Initialize the exception.

        Args:
            provider: Name of the provider.
            reason: Optional explanation of why streaming isn't supported.
        """
        self.provider = provider
        self.reason = reason
        message = f"Streaming not supported for {provider}"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class FileUploadNotSupportedError(LLMClientError):
    """Raised when file upload is requested but not supported.

    This exception is raised when attempting to upload files to a provider
    that doesn't support file/multimodal inputs.

    Attributes:
        provider: Name of the provider.
        file_type: Type of file that was attempted to upload.

    Examples:
        >>> raise FileUploadNotSupportedError("groq", "pdf")
        FileUploadNotSupportedError: File upload not supported for groq provider.
        Supported file type: pdf
    """

    def __init__(self, provider: str, file_type: str | None = None):
        """Initialize the exception.

        Args:
            provider: Name of the provider.
            file_type: Optional file type that was attempted.
        """
        self.provider = provider
        self.file_type = file_type
        message = f"File upload not supported for {provider} provider"
        if file_type:
            message += f" (file type: {file_type})"
        super().__init__(message)
