"""Extended tests for LLMClient to increase coverage."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_client import LLMClient, setup_logging


class TestGoogleColabIntegration:
    """Tests for Google Colab integration."""

    def setup_method(self):
        """Set logging to DEBUG for better coverage."""
        setup_logging(level="DEBUG", force=True)

    def test_load_colab_secrets_when_in_colab(self, monkeypatch):
        """Test: Load secrets from Colab userdata."""
        # Simulate Colab environment
        monkeypatch.setenv("COLAB_GPU", "1")

        mock_userdata = MagicMock()
        mock_userdata.get.side_effect = lambda key: {
            "OPENAI_API_KEY": "sk-colab-key",
            "GROQ_API_KEY": None,
            "GEMINI_API_KEY": None,
        }.get(key)

        mock_colab = MagicMock()
        mock_colab.userdata = mock_userdata

        with (
            patch.dict("sys.modules", {"google.colab": mock_colab}),
            patch("llm_client.providers.providers.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()

            client = LLMClient()

            assert client.openai_api_key == "sk-colab-key"

    def test_load_colab_secrets_specific_provider(self, monkeypatch):
        """Test: Load only specific provider key from Colab."""
        monkeypatch.setenv("COLAB_GPU", "1")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        mock_userdata = MagicMock()
        mock_userdata.get.return_value = "AIzaSy-colab-key"

        mock_colab = MagicMock()
        mock_colab.userdata = mock_userdata

        with (
            patch.dict("sys.modules", {"google.colab": mock_colab}),
            patch("llm_client.providers.providers.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="gemini")

            assert client.gemini_api_key == "AIzaSy-colab-key"

    def test_colab_secrets_handles_exceptions(self, monkeypatch):
        """Test: Handles exceptions when loading Colab secrets."""
        monkeypatch.setenv("COLAB_GPU", "1")

        mock_userdata = MagicMock()
        mock_userdata.get.side_effect = Exception("Colab error")

        mock_colab = MagicMock()
        mock_colab.userdata = mock_userdata

        with patch.dict("sys.modules", {"google.colab": mock_colab}):
            # Should not crash, just print error
            client = LLMClient(api_choice="ollama")

            assert client.api_choice == "ollama"

    def test_no_colab_loading_when_not_in_colab(self, monkeypatch):
        """Test: Doesn't try to load Colab secrets outside Colab."""
        monkeypatch.delenv("COLAB_GPU", raising=False)

        if "google.colab" in sys.modules:
            del sys.modules["google.colab"]

        client = LLMClient(api_choice="ollama")

        # Should work without trying to access Colab
        assert client.api_choice == "ollama"

    def test_colab_switch_provider_loads_key(self, monkeypatch):
        """Test: Switching provider in Colab loads new key."""
        monkeypatch.setenv("COLAB_GPU", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        mock_userdata = MagicMock()
        mock_userdata.get.return_value = "gsk-colab-key"

        mock_colab = MagicMock()
        mock_colab.userdata = mock_userdata

        with (
            patch.dict("sys.modules", {"google.colab": mock_colab}),
            patch("llm_client.providers.providers.OpenAI") as mock_openai,
            patch("llm_client.providers.providers.Groq") as mock_groq,
        ):
            mock_openai.return_value = MagicMock()
            mock_groq.return_value = MagicMock()

            client = LLMClient(api_choice="openai")

            # Switch to Groq - should load Groq key from Colab
            client.switch_provider("groq")

            assert client.groq_api_key == "gsk-colab-key"

    def test_colab_secrets_auto_selection(self, monkeypatch):
        """Test: Load all available keys from Colab for auto-selection."""
        monkeypatch.setenv("COLAB_GPU", "1")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        mock_userdata = MagicMock()
        mock_userdata.get.side_effect = lambda key: "key-for-" + key

        mock_colab = MagicMock()
        mock_colab.userdata = mock_userdata

        with (
            patch.dict("sys.modules", {"google.colab": mock_colab}),
            patch("llm_client.providers.providers.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            # api_choice is None for auto-selection
            client = LLMClient(api_choice=None)
            assert client.openai_api_key == "key-for-OPENAI_API_KEY"
            assert client.groq_api_key == "key-for-GROQ_API_KEY"


class TestLLMClientFromConfig:
    """Tests for from_config method."""

    def test_from_config_invalid_config(self, tmp_path, monkeypatch):
        """Test: Raises ValueError for invalid config."""
        pytest.importorskip("yaml")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("""
default_provider: nonexistent
providers:
  openai:
    model: gpt-4o
""")

        with pytest.raises(ValueError, match="Invalid configuration"):
            LLMClient.from_config(config_path)

    def test_from_config_with_pathlib(self, tmp_path, monkeypatch):
        """Test: from_config accepts pathlib.Path."""
        pytest.importorskip("yaml")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        from llm_client.config import generate_config_template

        config_path = tmp_path / "config.yaml"
        generate_config_template(config_path)

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient.from_config(Path(config_path))

            assert client is not None

    def test_from_config_uses_keep_alive(self, tmp_path):
        """Test: from_config uses keep_alive from config."""
        pytest.importorskip("yaml")

        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
default_provider: ollama
providers:
  ollama:
    model: llama3.2:1b
    keep_alive: 15m
""")
        mock_response = {"message": {"content": "test"}}

        with patch("llm_client.providers.providers.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.chat.return_value = mock_response
            mock_client.return_value = mock_instance

            client = LLMClient.from_config(config_path)

            assert client.keep_alive == "15m"

    def test_from_config_async_client(self, tmp_path, monkeypatch):
        """Test: Create async client from config."""
        pytest.importorskip("yaml")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        from llm_client.config import generate_config_template

        config_path = tmp_path / "config.yaml"
        generate_config_template(config_path)

        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_async_openai.return_value = MagicMock()

            client = LLMClient.from_config(config_path, use_async=True)

            assert client.use_async is True


class TestLLMClientProperties:
    """Tests for LLMClient properties."""

    def test_llm_property(self, monkeypatch):
        """Test: llm property returns current model."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai", llm="gpt-4o")

            assert client.llm == "gpt-4o"

    def test_client_property_backward_compatibility(self, monkeypatch):
        """Test: client property for backward compatibility."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")

            assert client.client == mock_client

    def test_repr_with_async(self, monkeypatch):
        """Test: repr includes async suffix."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_async_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai", use_async=True)

            repr_str = repr(client)
            assert "async" in repr_str.lower()


class TestLLMClientAsyncMethods:
    """Tests for async method error handling."""

    def test_achat_completion_with_sync_provider(self, monkeypatch):
        """Test: Sync provider raises error for async methods."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai", use_async=False)

            import asyncio

            with pytest.raises(RuntimeError, match="does not support async"):
                asyncio.run(client.achat_completion([]))

    def test_achat_completion_with_tools_sync_provider(self, monkeypatch):
        """Test: Sync provider raises error for async tool calling."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai", use_async=False)

            import asyncio

            with pytest.raises(RuntimeError, match="does not support async"):
                asyncio.run(client.achat_completion_with_tools([], []))

    def test_achat_completion_stream_sync_provider(self, monkeypatch):
        """Test: Sync provider raises error for async streaming."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai", use_async=False)

            import asyncio

            async def test():
                with pytest.raises(RuntimeError, match="does not support async"):
                    async for _ in client.achat_completion_stream([]):
                        pass

            asyncio.run(test())


class TestLLMClientInitializationEdgeCases:
    """Tests for edge cases in initialization."""

    def test_initialization_with_secrets_file(self, tmp_path, monkeypatch):
        """Test: Load secrets from custom secrets file."""
        secrets_file = tmp_path / "custom.env"
        secrets_file.write_text("OPENAI_API_KEY=sk-custom")

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(secrets_path=str(secrets_file))

            assert client.openai_api_key == "sk-custom"

    def test_initialization_keep_alive_parameter(self):
        """Test: Initialize with custom keep_alive."""
        client = LLMClient(api_choice="ollama", keep_alive="20m")

        assert client.keep_alive == "20m"

    def test_get_api_choice_from_provider(self, monkeypatch):
        """Test: _get_api_choice_from_provider method."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai")

            # Should infer from provider class name
            assert client.api_choice == "openai"

    def test_initialization_with_all_parameters(self, monkeypatch):
        """Test: Initialize with all parameters."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(
                llm="gpt-4o",
                temperature=0.5,
                max_tokens=2048,
                api_choice="openai",
                secrets_path="secrets.env",
                keep_alive="10m",
                use_async=False,
            )

            assert client.llm == "gpt-4o"
            assert client.temperature == 0.5
            assert client.max_tokens == 2048
            assert client.keep_alive == "10m"
            assert client.use_async is False


class TestLLMClientTokenCounting:
    """Tests for token counting methods."""

    def test_count_tokens_with_custom_model(self, monkeypatch):
        """Test: Count tokens with specific model."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with (
            patch("llm_client.providers.providers.OpenAI") as mock_openai,
            patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", False),
        ):
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai", llm="gpt-4o")
            messages = [{"role": "user", "content": "Test"}]

            # Count with different model
            count = client.count_tokens(messages, model="gpt-3.5-turbo")

            assert isinstance(count, int)
            assert count > 0

    def test_count_string_tokens(self, monkeypatch):
        """Test: Count tokens in string."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with (
            patch("llm_client.providers.providers.OpenAI") as mock_openai,
            patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", False),
        ):
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai")

            count = client.count_string_tokens("Hello, world!")

            assert isinstance(count, int)
            assert count > 0

    def test_count_string_tokens_with_custom_model(self, monkeypatch):
        """Test: Count string tokens with specific model."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with (
            patch("llm_client.providers.providers.OpenAI") as mock_openai,
            patch("llm_client.utils.token_counter.TIKTOKEN_AVAILABLE", False),
        ):
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai", llm="gpt-4o")

            count = client.count_string_tokens("Test", model="gpt-3.5-turbo")

            assert isinstance(count, int)


class TestLLMClientChatCompletionWithTools:
    """Tests for chat_completion_with_tools method."""

    def test_chat_completion_with_tools(self, monkeypatch):
        """Test: chat_completion_with_tools method."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "NYC"}'

        mock_response = MagicMock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = [mock_tool_call]

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")

            messages = [{"role": "user", "content": "Weather?"}]
            tools = [{"type": "function", "function": {"name": "get_weather"}}]

            result = client.chat_completion_with_tools(messages, tools)

            assert result["tool_calls"] is not None
            assert len(result["tool_calls"]) == 1

    def test_chat_completion_with_tools_and_tool_choice(self, monkeypatch):
        """Test: chat_completion_with_tools with tool_choice parameter."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = None

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")

            messages = [{"role": "user", "content": "Test"}]
            tools = [{"type": "function", "function": {"name": "test"}}]
            tool_choice = {"type": "function", "function": {"name": "test"}}

            result = client.chat_completion_with_tools(messages, tools, tool_choice)

            assert result["content"] == "Response"


class TestLLMClientStreamingMethods:
    """Tests for streaming methods."""

    def test_chat_completion_stream(self, monkeypatch):
        """Test: chat_completion_stream method."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))]),
        ]

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = iter(mock_chunks)
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Test"}]

            chunks = list(client.chat_completion_stream(messages))

            assert chunks == ["Hello", " world"]


class TestLLMClientAdditionalCoverage:
    """Extra tests to reach 95%+."""

    def test_llm_client_init_key_logs(self, monkeypatch):
        """Test: All API key logging branches in __init__."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama-test")
        monkeypatch.setenv("KICONNECT_API_KEY", "kiconnect-test")
        monkeypatch.setenv("API_KEY", "generic-test")

        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            client = LLMClient()
            assert client.openai_api_key == "sk-test"
            assert client.groq_api_key == "gsk-test"
            assert client.gemini_api_key == "AIzaSy-test"
            assert client.ollama_api_key == "ollama-test"
            assert client.kiconnect_api_key == "kiconnect-test"
            assert client.api_key == "generic-test"

    def test_get_api_choice_from_provider_branches(self, monkeypatch):
        """Test: All branches of _get_api_choice_from_provider."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        monkeypatch.setenv("GROQ_API_KEY", "gsk")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza")

        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            # Groq
            with patch("llm_client.providers.providers.Groq", MagicMock()):
                client = LLMClient(api_choice="groq")
                assert client._get_api_choice_from_provider() == "groq"

            # Gemini
            client = LLMClient(api_choice="gemini")
            assert client._get_api_choice_from_provider() == "gemini"

            # Ollama
            with patch("llm_client.providers.providers.Client", MagicMock()):
                client = LLMClient(api_choice="ollama")
                assert client._get_api_choice_from_provider() == "ollama"

            # Unknown
            client = LLMClient(api_choice="openai")
            client.provider = MagicMock()
            client.provider.__class__.__name__ = "UnknownProvider"
            assert client._get_api_choice_from_provider() == "unknown"

    @pytest.mark.asyncio
    async def test_achat_completion_stream(self, monkeypatch):
        """Test: achat_completion_stream method."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        async def mock_stream_gen(msgs):
            yield "chunk1"
            yield "chunk2"

        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_client = MagicMock()
            mock_client.achat_completion_stream.return_value = mock_stream_gen([])
            mock_async_openai.return_value = mock_client

            client = LLMClient(api_choice="openai", use_async=True)
            client.provider = mock_client

            chunks = []
            async for chunk in client.achat_completion_stream([]):
                chunks.append(chunk)

            assert chunks == ["chunk1", "chunk2"]

    def test_count_string_tokens_direct(self, monkeypatch):
        """Test: count_string_tokens implementation."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            client = LLMClient(api_choice="openai")
            count = client.count_string_tokens("hello")
            assert count > 0

    def test_chat_completion_with_files_validation_error(self, monkeypatch):
        """Test: File not found error in chat_completion_with_files."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            client = LLMClient(api_choice="openai")
            with pytest.raises(FileNotFoundError, match="File not found: nonexistent.jpg"):
                client.chat_completion_with_files([], files=["nonexistent.jpg"])

    def test_logging_chat_completion_with_files(self, monkeypatch, tmp_path):
        """Test: Logging branches in chat_completion_with_files."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        test_file = tmp_path / "test.jpg"
        test_file.write_text("fake image data")

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value.choices[0].message.content = "Response"
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            client.chat_completion_with_files(
                [{"role": "user", "content": "hi"}], files=[str(test_file)]
            )

    @pytest.mark.asyncio
    async def test_achat_completion_with_files_no_async_support(self, monkeypatch):
        """Test: Error when async provider doesn't support files."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("llm_client.providers.async_providers.AsyncOpenAI", MagicMock()):
            client = LLMClient(api_choice="openai", use_async=True)
            # Replace provider with one that doesn't have the method
            client.provider = MagicMock(spec=object)

            with pytest.raises(RuntimeError, match="does not support async file uploads"):
                await client.achat_completion_with_files([], files=None)

    @pytest.mark.asyncio
    async def test_achat_completion_no_async_support(self, monkeypatch):
        """Test: Error when async provider doesn't support achat_completion."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            client = LLMClient(api_choice="openai", use_async=False)
            client.provider = MagicMock(spec=object)
            with pytest.raises(RuntimeError, match="does not support async"):
                await client.achat_completion([])

    @pytest.mark.asyncio
    async def test_achat_completion_with_tools_no_async_support(self, monkeypatch):
        """Test: Error when async provider doesn't support tools."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            client = LLMClient(api_choice="openai", use_async=False)
            client.provider = MagicMock(spec=object)
            with pytest.raises(RuntimeError, match="does not support async tools"):
                await client.achat_completion_with_tools([], [])

    def test_switch_provider_params_logging(self, monkeypatch):
        """Test: switch_provider parameter branches."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        monkeypatch.setenv("GROQ_API_KEY", "gsk")
        with (
            patch("llm_client.providers.providers.OpenAI", MagicMock()),
            patch("llm_client.providers.providers.Groq", MagicMock()),
        ):
            client = LLMClient(api_choice="openai")
            client.switch_provider("groq", temperature=0.1, max_tokens=100, use_ollama_cloud=True)
            assert client.temperature == 0.1
            assert client.max_tokens == 100
            assert client.use_ollama_cloud is True

    def test_load_colab_secrets_unfilled_only(self, monkeypatch):
        """Test: colab secrets only fill empty keys."""
        monkeypatch.setenv("COLAB_GPU", "1")
        mock_userdata = MagicMock()
        mock_userdata.get.return_value = "colab-key"
        with (
            patch.dict("sys.modules", {"google.colab": MagicMock(userdata=mock_userdata)}),
            patch("llm_client.providers.providers.OpenAI", MagicMock()),
        ):
            client = LLMClient(api_choice="openai")
            # Force keys to be None
            client.openai_api_key = None
            client.groq_api_key = None
            client._load_colab_secrets()
            assert client.openai_api_key == "colab-key"
            assert client.groq_api_key == "colab-key"

    def test_load_colab_secrets_exception_auto(self, monkeypatch):
        """Test: colab secrets handles exception during auto-selection."""
        monkeypatch.setenv("COLAB_GPU", "1")
        mock_userdata = MagicMock()
        mock_userdata.get.side_effect = Exception("Colab error")
        with (
            patch.dict("sys.modules", {"google.colab": MagicMock(userdata=mock_userdata)}),
            patch("llm_client.providers.providers.OpenAI", MagicMock()),
        ):
            client = LLMClient(api_choice="openai")
            client.openai_api_key = None
            # This should hit the except branch in the loop
            client._load_colab_secrets()
            assert client.openai_api_key is None

    def test_load_colab_secrets_outer_exception(self, monkeypatch):
        """Test: colab secrets handles outer exception when google.colab cannot be imported/accessed."""
        monkeypatch.setenv("COLAB_GPU", "1")

        class BadColab:
            @property
            def userdata(self):
                raise Exception("Fatal colab error")

        with (
            patch.dict("sys.modules", {"google.colab": BadColab()}),
            patch("llm_client.providers.providers.OpenAI", MagicMock()),
        ):
            client = LLMClient(api_choice="openai")
            assert client.api_choice == "openai"

    def test_get_api_choice_from_provider_kiconnect(self, monkeypatch):
        """Test: _get_api_choice_from_provider returns kiconnect."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk")
        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            client = LLMClient(api_choice="openai")
            client.provider = MagicMock()
            client.provider.__class__.__name__ = "KIConnectProvider"
            assert client._get_api_choice_from_provider() == "kiconnect"

    @pytest.mark.asyncio
    async def test_achat_completion_with_tools_success(self, monkeypatch):
        """Test: achat_completion_with_tools successful call."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_async_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai", use_async=True)

            async def mock_async_call(*args, **kwargs):
                return {"content": "tool result"}
            client.provider.achat_completion_with_tools = mock_async_call

            result = await client.achat_completion_with_tools([], [])
            assert result == {"content": "tool result"}

    @pytest.mark.asyncio
    async def test_achat_completion_with_files_nonexistent_raises_error(self, monkeypatch):
        """Test: achat_completion_with_files raises FileNotFoundError for missing files."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_provider = MagicMock()
            mock_provider.achat_completion_with_files = MagicMock()
            mock_async_openai.return_value = mock_provider

            client = LLMClient(api_choice="openai", use_async=True)
            with pytest.raises(FileNotFoundError, match="File not found: nonexistent_file.txt"):
                await client.achat_completion_with_files([], files=["nonexistent_file.txt"])
