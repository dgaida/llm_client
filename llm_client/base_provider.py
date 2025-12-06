"""Base provider interface for LLM clients."""

from abc import ABC, abstractmethod
from typing import Any


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
            llm: Model name to use.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
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
            RuntimeError: If client initialization fails.
        """
        pass

    @abstractmethod
    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        """Execute a chat completion request.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.

        Returns:
            The generated text response.

        Raises:
            RuntimeError: If the API call fails.
        """
        pass

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
