"""Comprehensive tests for new features: token counting, async, and config files."""

import json
from unittest.mock import MagicMock, patch

import pytest

from llm_client import LLMClient
from llm_client.config import LLMConfig, create_default_config, generate_config_template
from llm_client.token_counter import TokenCounter


class TestTokenCounting:
    """Tests for token counting functionality."""

    def test_token_counter_with_tiktoken(self):
        """Test: Token counting with tiktoken installed."""
        if not TokenCounter.is_tiktoken_available():
            pytest.skip("tiktoken not available")

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello world!"},
        ]

        token_count = TokenCounter.count_tokens(messages, model="gpt-4o")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_token_counter_fallback(self):
        """Test: Token counting falls back to estimation."""
        messages = [
            {"role": "user", "content": "Hello world!"},
        ]

        # Should work even without tiktoken (uses estimation)
        token_count = TokenCounter.count_tokens(messages, model="gpt-4o", fallback=True)
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_token_counter_no_fallback_raises_error(self):
        """Test: No fallback raises ImportError if tiktoken unavailable."""
        if TokenCounter.is_tiktoken_available():
            pytest.skip("tiktoken is available")

        with pytest.raises(ImportError, match="tiktoken is required"):
            TokenCounter.count_tokens(
                [{"role": "user", "content": "test"}], model="gpt-4o", fallback=False
            )

    def test_count_string_tokens(self):
        """Test: Count tokens in a string."""
        text = "Hello, how are you?"
        token_count = TokenCounter.count_string_tokens(text, model="gpt-4o")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_different_models(self):
        """Test: Token counting for different models."""
        text = "Test message"
        models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

        for model in models:
            count = TokenCounter.count_string_tokens(text, model=model)
            assert isinstance(count, int)
            assert count > 0

    def test_llm_client_count_tokens(self, monkeypatch):
        """Test: LLMClient.count_tokens method."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Hello"}]

            token_count = client.count_tokens(messages)
            assert isinstance(token_count, int)
            assert token_count > 0

    def test_llm_client_count_string_tokens(self, monkeypatch):
        """Test: LLMClient.count_string_tokens method."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai")
            token_count = client.count_string_tokens("Hello world!")

            assert isinstance(token_count, int)
            assert token_count > 0

    def test_token_counting_empty_messages(self):
        """Test: Token counting with empty messages."""
        messages = []
        token_count = TokenCounter.count_tokens(messages, model="gpt-4o")
        assert isinstance(token_count, int)

    def test_token_counting_with_custom_model(self, monkeypatch):
        """Test: Token counting with custom model name."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai", llm="gpt-4o")
            messages = [{"role": "user", "content": "Test"}]

            # Count with client's model
            count1 = client.count_tokens(messages)

            # Count with different model
            count2 = client.count_tokens(messages, model="gpt-3.5-turbo")

            assert isinstance(count1, int)
            assert isinstance(count2, int)


class TestConfigFileSupport:
    """Tests for configuration file support."""

    def test_create_config_from_dict(self):
        """Test: Create config from dictionary."""
        config_dict = {
            "default_provider": "openai",
            "providers": {
                "openai": {"model": "gpt-4o", "temperature": 0.7},
            },
        }

        config = LLMConfig.from_dict(config_dict)
        assert config.default_provider == "openai"
        assert "openai" in config.providers

    def test_create_config_from_yaml(self, tmp_path):
        """Test: Create config from YAML file."""
        pytest.importorskip("yaml")

        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(
            """
default_provider: groq
providers:
  groq:
    model: llama-3.3-70b-versatile
    temperature: 0.5
"""
        )

        config = LLMConfig.from_file(config_path)
        assert config.default_provider == "groq"
        assert "groq" in config.providers

    def test_create_config_from_json(self, tmp_path):
        """Test: Create config from JSON file."""
        config_dict = {
            "default_provider": "gemini",
            "providers": {"gemini": {"model": "gemini-2.5-flash", "temperature": 0.8}},
        }

        config_path = tmp_path / "test_config.json"
        config_path.write_text(json.dumps(config_dict))

        config = LLMConfig.from_file(config_path)
        assert config.default_provider == "gemini"

    def test_config_file_not_found(self):
        """Test: FileNotFoundError for missing config file."""
        with pytest.raises(FileNotFoundError):
            LLMConfig.from_file("nonexistent_config.yaml")

    def test_config_unsupported_format(self, tmp_path):
        """Test: ValueError for unsupported file format."""
        config_path = tmp_path / "config.txt"
        config_path.write_text("some text")

        with pytest.raises(ValueError, match="Unsupported file format"):
            LLMConfig.from_file(config_path)

    def test_config_get_provider_config(self):
        """Test: Get provider configuration."""
        config = LLMConfig.from_dict(create_default_config())

        openai_config = config.get_provider_config("openai")
        assert "model" in openai_config
        assert openai_config["model"] == "gpt-4o-mini"

    def test_config_get_nonexistent_provider(self):
        """Test: KeyError for nonexistent provider."""
        config = LLMConfig.from_dict(create_default_config())

        with pytest.raises(KeyError, match="Provider 'invalid' not found"):
            config.get_provider_config("invalid")

    def test_config_list_providers(self):
        """Test: List all providers."""
        config = LLMConfig.from_dict(create_default_config())
        providers = config.list_providers()

        assert "openai" in providers
        assert "groq" in providers
        assert "gemini" in providers
        assert "ollama" in providers

    def test_config_validation_success(self):
        """Test: Valid configuration passes validation."""
        config = LLMConfig.from_dict(create_default_config())
        is_valid, errors = config.validate()

        assert is_valid is True
        assert len(errors) == 0

    def test_config_validation_missing_default_provider(self):
        """Test: Validation fails for missing default provider."""
        config_dict = {
            "default_provider": "nonexistent",
            "providers": {"openai": {"model": "gpt-4o"}},
        }

        config = LLMConfig.from_dict(config_dict)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert len(errors) > 0

    def test_config_validation_missing_model(self):
        """Test: Validation fails for missing model field."""
        config_dict = {
            "default_provider": "openai",
            "providers": {"openai": {"temperature": 0.7}},  # Missing 'model'
        }

        config = LLMConfig.from_dict(config_dict)
        is_valid, errors = config.validate()

        assert is_valid is False
        assert "model" in str(errors)

    def test_config_save_to_file(self, tmp_path):
        """Test: Save configuration to file."""
        pytest.importorskip("yaml")

        config = LLMConfig.from_dict(create_default_config())
        output_path = tmp_path / "output_config.yaml"

        config.to_file(output_path)
        assert output_path.exists()

        # Verify can be loaded back
        loaded_config = LLMConfig.from_file(output_path)
        assert loaded_config.default_provider == config.default_provider

    def test_generate_config_template(self, tmp_path):
        """Test: Generate configuration template."""
        pytest.importorskip("yaml")

        output_path = tmp_path / "template.yaml"
        generate_config_template(output_path, format="yaml")

        assert output_path.exists()

        # Verify it's valid
        config = LLMConfig.from_file(output_path)
        assert config.default_provider is not None

    def test_llm_client_from_config(self, tmp_path, monkeypatch):
        """Test: Create LLMClient from config file."""
        pytest.importorskip("yaml")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        # Create config file
        config_path = tmp_path / "test.yaml"
        generate_config_template(config_path)

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            # Load from config
            client = LLMClient.from_config(config_path)

            assert client.api_choice == "openai"
            assert client.llm == "gpt-4o-mini"

    def test_llm_client_from_config_specific_provider(self, tmp_path, monkeypatch):
        """Test: Load specific provider from config."""
        pytest.importorskip("yaml")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        config_path = tmp_path / "test.yaml"
        generate_config_template(config_path)

        with patch("llm_client.providers.Groq") as mock_groq:
            mock_groq.return_value = MagicMock()

            client = LLMClient.from_config(config_path, provider="groq")

            assert client.api_choice == "groq"

    def test_config_merge_with_defaults(self):
        """Test: Merge config with default values."""
        config = LLMConfig.from_dict(
            {
                "default_provider": "openai",
                "providers": {"openai": {"model": "gpt-4o"}},
            }
        )

        defaults = {"temperature": 0.7, "max_tokens": 512}
        merged = config.merge_with_defaults(defaults)

        assert merged["model"] == "gpt-4o"
        assert merged["temperature"] == 0.7
        assert merged["max_tokens"] == 512


class TestAsyncSupport:
    """Tests for async support."""

    @pytest.mark.asyncio
    async def test_async_client_creation(self, monkeypatch):
        """Test: Create async LLMClient."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_async_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai", use_async=True)

            assert client.use_async is True
            assert "async" in repr(client).lower()

    @pytest.mark.asyncio
    async def test_async_chat_completion(self, monkeypatch):
        """Test: Async chat completion."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Async response"

        with patch("llm_client.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(return_value=mock_response)
            mock_async_openai.return_value = mock_client

            # Make the create method async
            async def async_create(*args, **kwargs):
                return mock_response

            mock_client.chat.completions.create = async_create

            client = LLMClient(api_choice="openai", use_async=True)
            messages = [{"role": "user", "content": "Hello"}]

            response = await client.achat_completion(messages)
            assert response == "Async response"

    def test_sync_client_cannot_use_async_methods(self, monkeypatch):
        """Test: Sync client raises error for async methods."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai", use_async=False)

            # Should raise error
            import asyncio

            with pytest.raises(RuntimeError, match="does not support async"):
                asyncio.run(client.achat_completion([{"role": "user", "content": "test"}]))


class TestIntegrationNewFeatures:
    """Integration tests combining multiple new features."""

    def test_config_with_token_counting(self, tmp_path, monkeypatch):
        """Test: Config loading + token counting."""
        pytest.importorskip("yaml")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        config_path = tmp_path / "config.yaml"
        generate_config_template(config_path)

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient.from_config(config_path)
            messages = [{"role": "user", "content": "Hello"}]

            # Should be able to count tokens
            token_count = client.count_tokens(messages)
            assert isinstance(token_count, int)

    @pytest.mark.asyncio
    async def test_async_from_config(self, tmp_path, monkeypatch):
        """Test: Create async client from config."""
        pytest.importorskip("yaml")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        config_path = tmp_path / "config.yaml"
        generate_config_template(config_path)

        with patch("llm_client.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_async_openai.return_value = MagicMock()

            client = LLMClient.from_config(config_path, use_async=True)

            assert client.use_async is True

    def test_all_features_together(self, tmp_path, monkeypatch):
        """Test: Token counting + config + switching providers."""
        pytest.importorskip("yaml")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        # Create config
        config_path = tmp_path / "config.yaml"
        generate_config_template(config_path)

        with (
            patch("llm_client.providers.OpenAI") as mock_openai,
            patch("llm_client.providers.Groq") as mock_groq,
        ):
            mock_openai.return_value = MagicMock()
            mock_groq.return_value = MagicMock()

            # Load from config
            client = LLMClient.from_config(config_path)
            messages = [{"role": "user", "content": "Hello"}]

            # Count tokens
            tokens1 = client.count_tokens(messages)
            assert isinstance(tokens1, int)

            # Switch provider
            client.switch_provider("groq")

            # Count tokens again
            tokens2 = client.count_tokens(messages)
            assert isinstance(tokens2, int)
