"""Token counting utilities for LLM messages using tiktoken."""

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class TokenCounter:
    """Utility class for counting tokens in messages.

    This class uses tiktoken to count tokens accurately for different models.
    Falls back to rough estimation if tiktoken is not available.

    Examples:
        >>> counter = TokenCounter()
        >>> messages = [{"role": "user", "content": "Hello world"}]
        >>> token_count = counter.count_tokens(messages, model="gpt-4o")
        >>> print(f"Tokens: {token_count}")
    """

    # Model to encoding mapping
    MODEL_ENCODINGS = {
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
    }

    @staticmethod
    def count_tokens(
        messages: list[dict[str, str]],
        model: str = "gpt-4o-mini",
        fallback: bool = True,
    ) -> int:
        """Count tokens in a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model name to use for encoding selection.
            fallback: If True, use rough estimation when tiktoken unavailable.

        Returns:
            Total token count.

        Raises:
            ImportError: If tiktoken not installed and fallback=False.

        Examples:
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "Hello!"}
            ... ]
            >>> TokenCounter.count_tokens(messages, model="gpt-4o")
            23
        """
        if not TIKTOKEN_AVAILABLE:
            if not fallback:
                raise ImportError(
                    "tiktoken is required for accurate token counting. "
                    "Install with: pip install tiktoken"
                )
            return TokenCounter._estimate_tokens(messages)

        try:
            encoding = TokenCounter._get_encoding(model)
            return TokenCounter._count_with_tiktoken(messages, encoding, model)
        except Exception:
            if fallback:
                return TokenCounter._estimate_tokens(messages)
            raise

    @staticmethod
    def _get_encoding(model: str) -> "tiktoken.Encoding":
        """Get tiktoken encoding for a model.

        Args:
            model: Model name.

        Returns:
            Tiktoken encoding instance.
        """
        # Try to get encoding from model name mapping
        encoding_name = TokenCounter.MODEL_ENCODINGS.get(model)

        if encoding_name:
            return tiktoken.get_encoding(encoding_name)

        # Try to get encoding directly for the model
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # Default to cl100k_base for unknown models
            return tiktoken.get_encoding("cl100k_base")

    @staticmethod
    def _count_with_tiktoken(
        messages: list[dict[str, str]],
        encoding: "tiktoken.Encoding",
        model: str,
    ) -> int:
        """Count tokens using tiktoken.

        Based on OpenAI's token counting example:
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        Args:
            messages: List of message dictionaries.
            encoding: Tiktoken encoding instance.
            model: Model name.

        Returns:
            Total token count.
        """
        # Tokens per message and per name (varies by model)
        if model.startswith("gpt-4o") or model.startswith("gpt-4") or model.startswith("gpt-3.5"):
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            # Default values
            tokens_per_message = 3
            tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name

        # Every reply is primed with <|start|>assistant<|message|>
        num_tokens += 3

        return num_tokens

    @staticmethod
    def _estimate_tokens(messages: list[dict[str, str]]) -> int:
        """Rough token estimation when tiktoken is not available.

        Uses approximation: 1 token ≈ 4 characters.

        Args:
            messages: List of message dictionaries.

        Returns:
            Estimated token count.
        """
        total_chars = sum(
            len(str(msg.get("role", ""))) + len(str(msg.get("content", ""))) for msg in messages
        )
        # Rough approximation: 1 token ≈ 4 characters, plus overhead
        return (total_chars // 4) + (len(messages) * 4)

    @staticmethod
    def count_string_tokens(text: str, model: str = "gpt-4o-mini") -> int:
        """Count tokens in a single string.

        Args:
            text: Text to count tokens for.
            model: Model name for encoding selection.

        Returns:
            Token count.

        Examples:
            >>> TokenCounter.count_string_tokens("Hello world!", "gpt-4o")
            3
        """
        if not TIKTOKEN_AVAILABLE:
            # Rough estimation
            return len(text) // 4

        encoding = TokenCounter._get_encoding(model)
        return len(encoding.encode(text))

    @staticmethod
    def is_tiktoken_available() -> bool:
        """Check if tiktoken is available.

        Returns:
            True if tiktoken is installed.
        """
        return TIKTOKEN_AVAILABLE
