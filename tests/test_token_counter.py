"""Comprehensive tests for TokenCounter module."""

from unittest.mock import MagicMock, patch

import pytest

from llm_client.utils.token_counter import TokenCounter


class TestTokenCounterAvailability:
    """Tests for tiktoken availability checking."""

    def test_is_tiktoken_available_when_installed(self):
        """Test: is_tiktoken_available returns True when tiktoken is installed."""
        with patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", True):
            assert TokenCounter.is_tiktoken_available() is True

    def test_is_tiktoken_available_when_not_installed(self):
        """Test: is_tiktoken_available returns False when tiktoken is not installed."""
        with patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            assert TokenCounter.is_tiktoken_available() is False


class TestTokenCountingWithTiktoken:
    """Tests for token counting with tiktoken available."""

    @pytest.fixture
    def mock_tiktoken(self):
        """Mock tiktoken module."""
        # Import the module to ensure tiktoken attribute exists
        import llm_client.utils.token_counter as token_counter_module

        # Create mock tiktoken if it doesn't exist
        if not hasattr(token_counter_module, "tiktoken"):
            token_counter_module.tiktoken = MagicMock()

        with (
            patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", True),
            patch.object(token_counter_module, "tiktoken") as mock_tk,
        ):
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_tk.get_encoding.return_value = mock_encoding
            mock_tk.encoding_for_model.return_value = mock_encoding
            yield mock_tk

    def test_count_tokens_gpt4o(self, mock_tiktoken):
        """Test: Count tokens for GPT-4o model."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        count = TokenCounter.count_tokens(messages, model="gpt-4o")

        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_gpt4o_mini(self, mock_tiktoken):
        """Test: Count tokens for GPT-4o-mini model."""
        messages = [{"role": "user", "content": "Test"}]

        count = TokenCounter.count_tokens(messages, model="gpt-4o-mini")

        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_gpt35_turbo(self, mock_tiktoken):
        """Test: Count tokens for GPT-3.5-turbo model."""
        messages = [{"role": "user", "content": "Test"}]

        count = TokenCounter.count_tokens(messages, model="gpt-3.5-turbo")

        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_with_name_field(self, mock_tiktoken):
        """Test: Count tokens with 'name' field in messages."""
        messages = [
            {"role": "system", "content": "You are helpful.", "name": "system"},
            {"role": "user", "content": "Hello!", "name": "user"},
        ]

        count = TokenCounter.count_tokens(messages, model="gpt-4o")

        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_unknown_model(self, mock_tiktoken):
        """Test: Count tokens for unknown model falls back to cl100k_base."""
        messages = [{"role": "user", "content": "Test"}]

        # Simulate unknown model
        mock_tiktoken.encoding_for_model.side_effect = KeyError("Unknown model")
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_tiktoken.get_encoding.return_value = mock_encoding

        count = TokenCounter.count_tokens(messages, model="unknown-model")

        assert isinstance(count, int)
        assert count > 0
        mock_tiktoken.get_encoding.assert_called_with("cl100k_base")

    def test_count_string_tokens_with_tiktoken(self, mock_tiktoken):
        """Test: Count tokens in a string with tiktoken."""
        text = "Hello, how are you?"

        count = TokenCounter.count_string_tokens(text, model="gpt-4o")

        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_empty_messages(self, mock_tiktoken):
        """Test: Count tokens for empty messages list."""
        messages = []

        count = TokenCounter.count_tokens(messages, model="gpt-4o")

        assert isinstance(count, int)
        # Should still return priming tokens (3)
        assert count >= 3

    def test_get_encoding_from_mapping(self, mock_tiktoken):
        """Test: Get encoding from MODEL_ENCODINGS mapping."""
        with (
            patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", True),
            patch("llm_client.utils.token_counter.tiktoken") as mock_tk,
        ):
            mock_encoding = MagicMock()
            mock_tk.get_encoding.return_value = mock_encoding

            encoding = TokenCounter._get_encoding("gpt-4o")

            mock_tk.get_encoding.assert_called_with("o200k_base")
            assert encoding == mock_encoding

    def test_get_encoding_direct(self, mock_tiktoken):
        """Test: Get encoding directly for model not in mapping."""
        with (
            patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", True),
            patch("llm_client.utils.token_counter.tiktoken") as mock_tk,
        ):
            mock_encoding = MagicMock()
            mock_tk.encoding_for_model.return_value = mock_encoding

            encoding = TokenCounter._get_encoding("some-other-model")

            assert encoding == mock_encoding

    def test_count_with_tiktoken_implementation(self, mock_tiktoken):
        """Test: _count_with_tiktoken implementation."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
        ]

        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]

        count = TokenCounter._count_with_tiktoken(messages, mock_encoding, "gpt-4o")

        assert isinstance(count, int)
        assert count > 0


class TestTokenCountingFallback:
    """Tests for fallback token estimation."""

    def test_count_tokens_without_tiktoken_with_fallback(self):
        """Test: Fallback estimation when tiktoken unavailable."""
        with patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello world!"},
            ]

            count = TokenCounter.count_tokens(messages, model="gpt-4o", fallback=True)

            assert isinstance(count, int)
            assert count > 0

    def test_count_tokens_without_tiktoken_no_fallback(self):
        """Test: Raises ImportError without fallback."""
        with patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            messages = [{"role": "user", "content": "Test"}]

            with pytest.raises(ImportError, match="tiktoken is required"):
                TokenCounter.count_tokens(messages, model="gpt-4o", fallback=False)

    def test_estimate_tokens_implementation(self):
        """Test: _estimate_tokens implementation."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
        ]

        count = TokenCounter._estimate_tokens(messages)

        assert isinstance(count, int)
        assert count > 0

    def test_estimate_tokens_empty_messages(self):
        """Test: Estimate tokens for empty messages."""
        count = TokenCounter._estimate_tokens([])

        assert isinstance(count, int)
        assert count >= 0

    def test_estimate_tokens_long_content(self):
        """Test: Estimate tokens for long content."""
        messages = [{"role": "user", "content": "x" * 1000}]

        count = TokenCounter._estimate_tokens(messages)

        assert isinstance(count, int)
        assert count > 200  # Roughly 1000/4 + overhead

    def test_count_string_tokens_without_tiktoken(self):
        """Test: Count string tokens without tiktoken."""
        with patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            text = "Hello, how are you doing today?"

            count = TokenCounter.count_string_tokens(text, model="gpt-4o")

            assert isinstance(count, int)
            assert count > 0


class TestTokenCountingEdgeCases:
    """Tests for edge cases in token counting."""

    def test_count_tokens_with_exception_and_fallback(self):
        """Test: Falls back to estimation on exception."""
        with (
            patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", True),
            patch("llm_client.utils.token_counter.tiktoken") as mock_tk,
        ):
            mock_tk.get_encoding.side_effect = Exception("Encoding error")

            messages = [{"role": "user", "content": "Test"}]

            # Should fall back to estimation
            count = TokenCounter.count_tokens(messages, model="gpt-4o", fallback=True)

            assert isinstance(count, int)
            assert count > 0

    def test_count_tokens_with_exception_no_fallback(self):
        """Test: Raises exception when fallback disabled."""
        with (
            patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", True),
            patch("llm_client.utils.token_counter.tiktoken") as mock_tk,
        ):
            mock_tk.get_encoding.side_effect = Exception("Encoding error")

            messages = [{"role": "user", "content": "Test"}]

            with pytest.raises(Exception, match="Encoding error"):
                TokenCounter.count_tokens(messages, model="gpt-4o", fallback=False)

    def test_tiktoken_import_error_on_module_load(self):
        """Test: TIKTOKEN_AVAILABLE is False when tiktoken is not installed."""
        import importlib
        import sys

        # Temporarily hide tiktoken
        with patch.dict(sys.modules, {"tiktoken": None}):
            # Reload module
            import llm_client.utils.token_counter

            importlib.reload(llm_client.utils.token_counter)

            # Assert TIKTOKEN_AVAILABLE is False
            assert llm_client.utils.token_counter.TIKTOKEN_AVAILABLE is False

        # Restore module back to normal state
        importlib.reload(llm_client.utils.token_counter)

    def test_count_tokens_missing_content(self):
        """Test: Handle messages with missing content."""
        with patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            messages = [
                {"role": "user"},  # Missing content
                {"role": "system", "content": "System"},
            ]

            count = TokenCounter.count_tokens(messages, fallback=True)

            assert isinstance(count, int)

    def test_count_string_tokens_empty_string(self):
        """Test: Count tokens in empty string."""
        with patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            count = TokenCounter.count_string_tokens("", model="gpt-4o")

            assert isinstance(count, int)
            assert count == 0

    def test_count_tokens_special_characters(self):
        """Test: Count tokens with special characters."""
        with patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            messages = [{"role": "user", "content": "Hello! 你好 🌍"}]

            count = TokenCounter.count_tokens(messages, fallback=True)

            assert isinstance(count, int)
            assert count > 0

    def test_count_tokens_very_long_message(self):
        """Test: Count tokens in very long message."""
        with patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            messages = [{"role": "user", "content": "x" * 10000}]

            count = TokenCounter.count_tokens(messages, fallback=True)

            assert isinstance(count, int)
            assert count > 1000


class TestModelEncodingMapping:
    """Tests for model encoding mappings."""

    def test_gpt4o_uses_o200k_base(self):
        """Test: GPT-4o uses o200k_base encoding."""
        with (
            patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", True),
            patch("llm_client.utils.token_counter.tiktoken") as mock_tk,
        ):
            mock_encoding = MagicMock()
            mock_tk.get_encoding.return_value = mock_encoding

            TokenCounter._get_encoding("gpt-4o")

            mock_tk.get_encoding.assert_called_with("o200k_base")

    def test_gpt4o_mini_uses_o200k_base(self):
        """Test: GPT-4o-mini uses o200k_base encoding."""
        with (
            patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", True),
            patch("llm_client.utils.token_counter.tiktoken") as mock_tk,
        ):
            mock_encoding = MagicMock()
            mock_tk.get_encoding.return_value = mock_encoding

            TokenCounter._get_encoding("gpt-4o-mini")

            mock_tk.get_encoding.assert_called_with("o200k_base")

    def test_gpt4_uses_cl100k_base(self):
        """Test: GPT-4 uses cl100k_base encoding."""
        with (
            patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", True),
            patch("llm_client.utils.token_counter.tiktoken") as mock_tk,
        ):
            mock_encoding = MagicMock()
            mock_tk.get_encoding.return_value = mock_encoding

            TokenCounter._get_encoding("gpt-4")

            mock_tk.get_encoding.assert_called_with("cl100k_base")

    def test_gpt35_turbo_uses_cl100k_base(self):
        """Test: GPT-3.5-turbo uses cl100k_base encoding."""
        with (
            patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", True),
            patch("llm_client.utils.token_counter.tiktoken") as mock_tk,
        ):
            mock_encoding = MagicMock()
            mock_tk.get_encoding.return_value = mock_encoding

            TokenCounter._get_encoding("gpt-3.5-turbo")

            mock_tk.get_encoding.assert_called_with("cl100k_base")

    def test_ada_uses_cl100k_base(self):
        """Test: text-embedding-ada-002 uses cl100k_base encoding."""
        with (
            patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", True),
            patch("llm_client.utils.token_counter.tiktoken") as mock_tk,
        ):
            mock_encoding = MagicMock()
            mock_tk.get_encoding.return_value = mock_encoding

            TokenCounter._get_encoding("text-embedding-ada-002")

            mock_tk.get_encoding.assert_called_with("cl100k_base")


class TestTokenCounterIntegration:
    """Integration tests for TokenCounter."""

    def test_multiple_models_consistency(self):
        """Test: Token counts are consistent across models."""
        with patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            messages = [{"role": "user", "content": "Test message"}]

            count1 = TokenCounter.count_tokens(messages, model="gpt-4o", fallback=True)
            count2 = TokenCounter.count_tokens(messages, model="gpt-4o-mini", fallback=True)
            count3 = TokenCounter.count_tokens(messages, model="gpt-3.5-turbo", fallback=True)

            # Should all be similar (using estimation)
            assert abs(count1 - count2) < 5
            assert abs(count2 - count3) < 5

    def test_count_increases_with_content_length(self):
        """Test: Token count increases with content length."""
        with patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            short_messages = [{"role": "user", "content": "Hi"}]
            long_messages = [{"role": "user", "content": "Hi " * 100}]

            short_count = TokenCounter.count_tokens(short_messages, fallback=True)
            long_count = TokenCounter.count_tokens(long_messages, fallback=True)

            assert long_count > short_count
