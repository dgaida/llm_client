"""Unit tests for provider implementations."""

from unittest.mock import MagicMock, patch

import pytest

from llm_client.exceptions import (
    APIKeyNotFoundError,
    ChatCompletionError,
    ProviderNotAvailableError,
)
from llm_client.providers.providers import (
    GeminiProvider,
    GroqProvider,
    OllamaProvider,
    OpenAIProvider,
)


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_initialization_success(self):
        """Test: OpenAI provider initializes correctly with valid API key."""
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(
                llm="gpt-4o", temperature=0.7, max_tokens=512, api_key="sk-test"
            )

            assert provider.llm == "gpt-4o"
            assert provider.temperature == 0.7
            assert provider.max_tokens == 512
            assert provider.client == mock_client
            mock_openai.assert_called_once_with(api_key="sk-test")

    def test_initialization_without_api_key_raises_error(self):
        """Test: RuntimeError when API key is missing."""
        with pytest.raises(
            APIKeyNotFoundError,
            match="OPENAI_API_KEY not found for openai provider. Please set it in environment or pass explicitly.",
        ):
            OpenAIProvider(llm="gpt-4o", temperature=0.7, max_tokens=512)

    def test_initialization_when_package_not_available(self):
        """Test: RuntimeError when OpenAI package is not installed."""
        with (
            patch("llm_client.providers.providers.OpenAI", None),
            pytest.raises(
                ProviderNotAvailableError,
                match="openai provider not available. Install with: pip install openai",
            ),
        ):
            OpenAIProvider(llm="gpt-4o", api_key="sk-test")

    def test_chat_completion_success(self):
        """Test: Chat completion returns correct response."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response"

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm="gpt-4o", api_key="sk-test")
            messages = [{"role": "user", "content": "Hello"}]
            response = provider.chat_completion(messages)

            assert response == "Test response"
            mock_client.chat.completions.create.assert_called_once_with(
                model="gpt-4o", messages=messages, temperature=0.7, max_tokens=512
            )

    def test_chat_completion_without_client_raises_error(self):
        """Test: RuntimeError when client is not initialized."""
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider = OpenAIProvider(llm="gpt-4o", api_key="sk-test")
            provider.client = None  # Simulate uninitialized client

            with pytest.raises(
                ChatCompletionError,
                match="Chat completion failed for OpenAIProvider provider: RuntimeError: OpenAI client not initialized",
            ):
                provider.chat_completion([{"role": "user", "content": "test"}])

    def test_get_default_model(self):
        """Test: Default model is correct."""
        assert OpenAIProvider.get_default_model() == "gpt-4o-mini"

    def test_is_available_when_installed(self):
        """Test: is_available returns True when package is installed."""
        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            assert OpenAIProvider.is_available() is True

    def test_is_available_when_not_installed(self):
        """Test: is_available returns False when package is not installed."""
        with patch("llm_client.providers.providers.OpenAI", None):
            assert OpenAIProvider.is_available() is False

    def test_repr(self):
        """Test: __repr__ returns correct string representation."""
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider = OpenAIProvider(
                llm="gpt-4o", temperature=0.5, max_tokens=1024, api_key="sk-test"
            )
            repr_str = repr(provider)

            assert "OpenAIProvider" in repr_str
            assert "gpt-4o" in repr_str
            assert "0.5" in repr_str

    def test_custom_parameters(self):
        """Test: Custom parameters are stored correctly."""
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider = OpenAIProvider(
                llm="gpt-3.5-turbo", temperature=0.3, max_tokens=2048, api_key="sk-test"
            )

            assert provider.llm == "gpt-3.5-turbo"
            assert provider.temperature == 0.3
            assert provider.max_tokens == 2048


class TestGroqProvider:
    """Tests for GroqProvider."""

    def test_initialization_success(self):
        """Test: Groq provider initializes correctly with valid API key."""
        with patch("llm_client.providers.providers.Groq") as mock_groq:
            mock_client = MagicMock()
            mock_groq.return_value = mock_client

            provider = GroqProvider(
                llm="llama-3.3-70b-versatile", temperature=0.7, max_tokens=512, api_key="gsk-test"
            )

            assert provider.llm == "llama-3.3-70b-versatile"
            assert provider.temperature == 0.7
            assert provider.max_tokens == 512
            assert provider.client == mock_client
            mock_groq.assert_called_once_with(api_key="gsk-test")

    def test_initialization_without_api_key_raises_error(self):
        """Test: APIKeyNotFoundError when API key is missing."""
        with pytest.raises(APIKeyNotFoundError, match="GROQ_API_KEY not found for groq provider"):
            GroqProvider(llm="llama-3.3-70b-versatile", temperature=0.7, max_tokens=512)

    def test_initialization_when_package_not_available(self):
        """Test: ProviderNotAvailableError when Groq package is not installed."""
        with (
            patch("llm_client.providers.providers.Groq", None),
            pytest.raises(
                ProviderNotAvailableError,
                match="groq provider not available. Install with: pip install groq",
            ),
        ):
            GroqProvider(llm="llama-3.3-70b-versatile", api_key="gsk-test")

    def test_chat_completion_success(self):
        """Test: Chat completion returns correct response."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Groq response"

        with patch("llm_client.providers.providers.Groq") as mock_groq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_groq.return_value = mock_client

            provider = GroqProvider(llm="llama-3.3-70b-versatile", api_key="gsk-test")
            messages = [{"role": "user", "content": "Hello"}]
            response = provider.chat_completion(messages)

            assert response == "Groq response"
            mock_client.chat.completions.create.assert_called_once()

    def test_chat_completion_without_client_raises_error(self):
        """Test: RuntimeError when client is not initialized."""
        with patch("llm_client.providers.providers.Groq") as mock_groq:
            mock_groq.return_value = MagicMock()

            provider = GroqProvider(llm="llama-3.3-70b-versatile", api_key="gsk-test")
            provider.client = None

            with pytest.raises(
                ChatCompletionError,
                match="Chat completion failed for GroqProvider provider: RuntimeError: Groq client not initialized",
            ):
                provider.chat_completion([{"role": "user", "content": "test"}])

    def test_get_default_model(self):
        """Test: Default model is correct."""
        assert GroqProvider.get_default_model() == "qwen/qwen3-32b"

    def test_is_available_when_installed(self):
        """Test: is_available returns True when package is installed."""
        with patch("llm_client.providers.providers.Groq", MagicMock()):
            assert GroqProvider.is_available() is True

    def test_is_available_when_not_installed(self):
        """Test: is_available returns False when package is not installed."""
        with patch("llm_client.providers.providers.Groq", None):
            assert GroqProvider.is_available() is False


class TestGeminiProvider:
    """Tests for GeminiProvider."""

    def test_initialization_success(self):
        """Test: Gemini provider initializes correctly with valid API key."""
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            provider = GeminiProvider(
                llm="gemini-2.5-flash", temperature=0.7, max_tokens=512, api_key="AIzaSy-test"
            )

            assert provider.llm == "gemini-2.5-flash"
            assert provider.temperature == 0.7
            assert provider.max_tokens == 512
            assert provider.client == mock_client
            mock_openai.assert_called_once_with(
                api_key="AIzaSy-test",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )

    def test_initialization_without_api_key_raises_error(self):
        """Test: APIKeyNotFoundError when API key is missing."""
        with pytest.raises(
            APIKeyNotFoundError, match="GEMINI_API_KEY not found for gemini provider"
        ):
            GeminiProvider(llm="gemini-2.5-flash", temperature=0.7, max_tokens=512)

    def test_initialization_when_package_not_available(self):
        """Test: ProviderNotAvailableError when OpenAI package (needed for Gemini) is not installed."""
        with (
            patch("llm_client.providers.providers.OpenAI", None),
            pytest.raises(
                ProviderNotAvailableError,
                match="gemini provider not available. Install with: pip install openai",
            ),
        ):
            GeminiProvider(llm="gemini-2.5-flash", api_key="AIzaSy-test")

    def test_chat_completion_success(self):
        """Test: Chat completion returns correct response."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Gemini response"

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = GeminiProvider(llm="gemini-2.5-flash", api_key="AIzaSy-test")
            messages = [{"role": "user", "content": "Hello"}]
            response = provider.chat_completion(messages)

            assert response == "Gemini response"
            mock_client.chat.completions.create.assert_called_once_with(
                model="gemini-2.5-flash", messages=messages, temperature=0.7, max_tokens=512
            )

    def test_chat_completion_without_client_raises_error(self):
        """Test: RuntimeError when client is not initialized."""
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider = GeminiProvider(llm="gemini-2.5-flash", api_key="AIzaSy-test")
            provider.client = None

            with pytest.raises(
                ChatCompletionError,
                match="Chat completion failed for GeminiProvider provider: RuntimeError: Gemini client not initialized",
            ):
                provider.chat_completion([{"role": "user", "content": "test"}])

    def test_get_default_model(self):
        """Test: Default model is correct."""
        assert GeminiProvider.get_default_model() == "gemini-3.1-flash-lite"

    def test_is_available_when_installed(self):
        """Test: is_available returns True when OpenAI package is installed."""
        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            assert GeminiProvider.is_available() is True

    def test_is_available_when_not_installed(self):
        """Test: is_available returns False when OpenAI package is not installed."""
        with patch("llm_client.providers.providers.OpenAI", None):
            assert GeminiProvider.is_available() is False

    def test_uses_correct_base_url(self):
        """Test: Gemini uses correct API base URL."""
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            GeminiProvider(llm="gemini-2.5-flash", api_key="AIzaSy-test")

            call_kwargs = mock_openai.call_args[1]
            assert (
                call_kwargs["base_url"]
                == "https://generativelanguage.googleapis.com/v1beta/openai/"
            )


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    def test_initialization_success(self):
        """Test: Ollama provider initializes correctly."""
        with patch("llm_client.providers.providers.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            provider = OllamaProvider(
                llm="llama3.2:1b", temperature=0.7, max_tokens=512, keep_alive="5m"
            )

            assert provider.llm == "llama3.2:1b"
            assert provider.temperature == 0.7
            assert provider.max_tokens == 512
            assert provider.keep_alive == "5m"
            assert provider.client == mock_instance  # Changed from: is None
            mock_client.assert_called_once()

    def test_initialization_when_package_not_available(self):
        """Test: ProviderNotAvailableError when ollama package is not installed."""
        with (
            patch("llm_client.providers.providers.Client", None),
            patch("llm_client.providers.providers.OLLAMA_AVAILABLE", False),
            pytest.raises(
                ProviderNotAvailableError,
                match="ollama provider not available. Install with: pip install ollama",
            ),
        ):
            OllamaProvider(llm="llama3.2:1b")

    def test_chat_completion_success(self):
        """Test: Chat completion returns correct response."""
        mock_response = {"message": {"content": "Ollama response"}}

        with patch("llm_client.providers.providers.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.chat.return_value = mock_response
            mock_client.return_value = mock_instance

            provider = OllamaProvider(llm="llama3.2:1b", keep_alive="10m")
            messages = [{"role": "user", "content": "Hello"}]
            response = provider.chat_completion(messages)

            assert response == "Ollama response"
            mock_instance.chat.assert_called_once_with(
                model="llama3.2:1b",
                messages=messages,
                stream=False,
                options={
                    "temperature": 0.7,
                    "num_predict": 512,
                    "repeat_penalty": 1.2,
                    "top_k": 10,
                    "top_p": 0.5,
                },
                keep_alive="10m",
            )

    def test_chat_completion_when_package_not_available(self):
        """Test: ProviderNotAvailableError when ollama package is not available during chat."""
        with patch("llm_client.providers.providers.Client", MagicMock()):
            provider = OllamaProvider(llm="llama3.2:1b")

        with (
            patch("llm_client.providers.providers.Client", None),
            patch("llm_client.providers.providers.OLLAMA_AVAILABLE", False),
            pytest.raises(
                ChatCompletionError,
                match="Chat completion failed for OllamaProvider provider: ProviderNotAvailableError: ollama provider not available. Install with: pip install ollama",
            ),
        ):
            provider.chat_completion([{"role": "user", "content": "test"}])

    def test_get_default_model(self):
        """Test: Default model is correct."""
        assert OllamaProvider.get_default_model() == "llama3.2:1b"

    def test_is_available_when_installed(self):
        """Test: is_available returns True when package is installed."""
        with patch("llm_client.providers.providers.Client", MagicMock()):
            assert OllamaProvider.is_available() is True

    def test_is_available_when_not_installed(self):
        """Test: is_available returns False when package is not installed."""
        with (
            patch("llm_client.providers.providers.Client", None),
            patch("llm_client.providers.providers.OLLAMA_AVAILABLE", False),
        ):
            assert OllamaProvider.is_available() is False

    def test_custom_keep_alive_parameter(self):
        """Test: Custom keep_alive parameter is stored correctly."""
        with patch("llm_client.providers.providers.Client", MagicMock()):
            provider = OllamaProvider(llm="llama3.2:1b", keep_alive="15m")
            assert provider.keep_alive == "15m"

    def test_custom_options_in_chat_completion(self):
        """Test: Custom parameters are passed correctly to ollama.chat."""
        mock_response = {"message": {"content": "Response"}}

        with patch("llm_client.providers.providers.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.chat.return_value = mock_response
            mock_client.return_value = mock_instance

            provider = OllamaProvider(llm="llama3.2:1b", temperature=0.5, max_tokens=1024)
            provider.chat_completion([{"role": "user", "content": "test"}])

            call_kwargs = mock_instance.chat.call_args[1]
            assert call_kwargs["options"]["temperature"] == 0.5
            assert call_kwargs["options"]["num_predict"] == 1024


class TestProviderEdgeCases:
    """Tests for edge cases across all providers."""

    def test_extreme_temperature_values(self):
        """Test: Providers handle extreme temperature values."""
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            # Very low temperature
            provider = OpenAIProvider(llm="gpt-4o", temperature=0.0, api_key="sk-test")
            assert provider.temperature == 0.0

            # Very high temperature
            provider = OpenAIProvider(llm="gpt-4o", temperature=2.0, api_key="sk-test")
            assert provider.temperature == 2.0

    def test_extreme_max_tokens_values(self):
        """Test: Providers handle extreme max_tokens values."""
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            # Very small
            provider = OpenAIProvider(llm="gpt-4o", max_tokens=1, api_key="sk-test")
            assert provider.max_tokens == 1

            # Very large
            provider = OpenAIProvider(llm="gpt-4o", max_tokens=100000, api_key="sk-test")
            assert provider.max_tokens == 100000

    def test_empty_messages_list(self):
        """Test: Providers handle empty messages list."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Response"

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm="gpt-4o", api_key="sk-test")
            response = provider.chat_completion([])

            assert response == "Response"

    def test_multiple_messages(self):
        """Test: Providers handle multiple messages correctly."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Final response"

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm="gpt-4o", api_key="sk-test")
            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
            ]
            response = provider.chat_completion(messages)

            assert response == "Final response"
            call_args = mock_client.chat.completions.create.call_args
            assert len(call_args[1]["messages"]) == 4


class TestProviderFeatures:
    """Tests for advanced provider features."""

    def test_openai_with_files(self):
        """Test: OpenAI provider handles file uploads."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "File response"

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm="gpt-4o", api_key="sk-test")

            with patch("llm_client.utils.file_utils.prepare_files_for_provider") as mock_prepare:
                mock_prepare.return_value = [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                ]

                messages = [{"role": "user", "content": "Analyze this"}]
                response = provider.chat_completion_with_files(messages, files=["test.jpg"])

                assert response == "File response"
                mock_client.chat.completions.create.assert_called_once()
                call_args = mock_client.chat.completions.create.call_args[1]
                assert isinstance(call_args["messages"][0]["content"], list)

    def test_openai_with_tools(self):
        """Test: OpenAI provider handles tool calls."""
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "Berlin"}'

        mock_response.choices[0].message.content = "Thinking..."
        mock_response.choices[0].message.tool_calls = [mock_tool_call]

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm="gpt-4o", api_key="sk-test")
            tools = [{"type": "function", "function": {"name": "get_weather"}}]

            result = provider.chat_completion_with_tools([], tools)

            assert result["content"] == "Thinking..."
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_ollama_cloud_mode(self):
        """Test: OllamaProvider in cloud mode."""
        with patch("llm_client.providers.providers.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            provider = OllamaProvider(
                llm="gpt-oss:120b-cloud",
                api_key="cloud-key",
                use_cloud=True,
                host="https://custom.ollama.com",
            )

            assert provider.use_cloud is True
            assert provider.host == "https://custom.ollama.com"
            mock_client.assert_called_once_with(
                host="https://custom.ollama.com", headers={"Authorization": "Bearer cloud-key"}
            )

    def test_ollama_with_files(self):
        """Test: OllamaProvider handles file uploads."""
        mock_response = {"message": {"content": "Ollama vision response"}}

        with patch("llm_client.providers.providers.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.chat.return_value = mock_response
            mock_client.return_value = mock_instance

            provider = OllamaProvider(llm="llava")

            with (
                patch("llm_client.utils.file_utils.detect_file_type", return_value="image"),
                patch("llm_client.utils.file_utils.encode_file_base64", return_value="base64data"),
            ):
                response = provider.chat_completion_with_files(
                    [{"role": "user", "content": "What is this?"}], files=["image.png"]
                )

                assert response == "Ollama vision response"
                mock_instance.chat.assert_called_once()
                call_kwargs = mock_instance.chat.call_args[1]
                assert "images" in call_kwargs["messages"][0]

    def test_groq_with_tools(self):
        """Test: GroqProvider handles tool calls."""
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_456"
        mock_tool_call.type = "function"
        mock_tool_call.function.name = "calculate"
        mock_tool_call.function.arguments = '{"expr": "2+2"}'

        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = [mock_tool_call]

        with patch("llm_client.providers.providers.Groq") as mock_groq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_groq.return_value = mock_client

            provider = GroqProvider(llm="llama3-groq-70b-8192-tool-use-preview", api_key="gsk-test")
            tools = [{"type": "function", "function": {"name": "calculate"}}]

            result = provider.chat_completion_with_tools([], tools)

            assert result["content"] is None
            assert result["tool_calls"][0]["function"]["name"] == "calculate"

    def test_gemini_with_tools(self):
        """Test: GeminiProvider handles tool calls."""
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_789"
        mock_tool_call.type = "function"
        mock_tool_call.function.name = "search"
        mock_tool_call.function.arguments = '{"query": "news"}'

        mock_response.choices[0].message.content = "Searching..."
        mock_response.choices[0].message.tool_calls = [mock_tool_call]

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = GeminiProvider(llm="gemini-3.1-flash-lite", api_key="AIzaSy-test")
            tools = [{"type": "function", "function": {"name": "search"}}]

            result = provider.chat_completion_with_tools([], tools, tool_choice="auto")

            assert result["content"] == "Searching..."
            assert result["tool_calls"][0]["function"]["name"] == "search"
            mock_client.chat.completions.create.assert_called_with(
                model="gemini-3.1-flash-lite",
                messages=[],
                temperature=0.7,
                max_tokens=512,
                tools=tools,
                tool_choice="auto",
            )

    def test_ollama_auto_cloud_mode(self):
        """Test: Ollama auto-detects cloud mode from model name."""
        with patch("llm_client.providers.providers.Client", MagicMock()):
            provider = OllamaProvider(llm="some-model-cloud", api_key="sk-key")
            assert provider.use_cloud is True

    def test_openai_with_tools_choice(self):
        """Test: OpenAI provider handles tool_choice."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Thinking..."
        mock_response.choices[0].message.tool_calls = None

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm="gpt-4o", api_key="sk-test")
            tools = [{"type": "function", "function": {"name": "get_weather"}}]

            provider.chat_completion_with_tools([], tools, tool_choice="required")
            mock_client.chat.completions.create.assert_called_with(
                model="gpt-4o",
                messages=[],
                temperature=0.7,
                max_tokens=512,
                tools=tools,
                tool_choice="required",
            )

    def test_groq_with_tools_choice(self):
        """Test: Groq provider handles tool_choice."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Thinking..."
        mock_response.choices[0].message.tool_calls = None

        with patch("llm_client.providers.providers.Groq") as mock_groq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_groq.return_value = mock_client

            provider = GroqProvider(llm="llama3", api_key="gsk-test")
            tools = [{"type": "function", "function": {"name": "get_weather"}}]

            provider.chat_completion_with_tools([], tools, tool_choice="none")
            mock_client.chat.completions.create.assert_called_with(
                model="llama3",
                messages=[],
                temperature=0.7,
                max_tokens=512,
                tools=tools,
                tool_choice="none",
            )

    def test_openai_streaming(self):
        """Test: OpenAI provider streaming."""
        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = "Test chunk"

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = [mock_chunk]
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm="gpt-4o", api_key="sk-test")
            chunks = list(provider.chat_completion_stream([{"role": "user", "content": "hi"}]))

            assert chunks == ["Test chunk"]

    def test_groq_streaming(self):
        """Test: Groq provider streaming."""
        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = "Groq chunk"

        with patch("llm_client.providers.providers.Groq") as mock_groq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = [mock_chunk]
            mock_groq.return_value = mock_client

            provider = GroqProvider(llm="llama3-8b", api_key="gsk-test")
            chunks = list(provider.chat_completion_stream([{"role": "user", "content": "hi"}]))

            assert chunks == ["Groq chunk"]

    def test_gemini_streaming(self):
        """Test: Gemini provider streaming."""
        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = "Gemini chunk"

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = [mock_chunk]
            mock_openai.return_value = mock_client

            provider = GeminiProvider(llm="gemini-2.0-flash", api_key="AIzaSy-test")
            chunks = list(provider.chat_completion_stream([{"role": "user", "content": "hi"}]))

            assert chunks == ["Gemini chunk"]

    def test_ollama_streaming(self):
        """Test: Ollama provider streaming."""
        mock_stream = [{"message": {"content": "O"}}, {"message": {"content": "k"}}]

        with patch("llm_client.providers.providers.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.chat.return_value = mock_stream
            mock_client.return_value = mock_instance

            provider = OllamaProvider(llm="llama3")
            chunks = list(provider.chat_completion_stream([{"role": "user", "content": "hi"}]))

            assert chunks == ["O", "k"]

    def test_groq_with_files(self):
        """Test: Groq provider with files (images)."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Groq vision"

        with patch("llm_client.providers.providers.Groq") as mock_groq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_groq.return_value = mock_client

            provider = GroqProvider(llm="llama-3.2-11b-vision-preview", api_key="gsk-test")

            with (
                patch("llm_client.utils.file_utils.detect_file_type", return_value="image"),
                patch(
                    "llm_client.utils.file_utils.prepare_files_for_provider",
                    return_value=[{"type": "image_url", "image_url": {"url": "..."}}],
                ),
            ):
                response = provider.chat_completion_with_files(
                    [{"role": "user", "content": "see"}], files=["img.jpg"]
                )
                assert response == "Groq vision"

    def test_groq_with_invalid_file_type_raises_error(self):
        """Test: Groq raises error for non-image files."""
        with patch("llm_client.providers.providers.Groq", MagicMock()):
            provider = GroqProvider(llm="llama-3.2-11b-vision-preview", api_key="gsk-test")

            with (
                patch("llm_client.utils.file_utils.detect_file_type", return_value="pdf"),
                pytest.raises(ChatCompletionError, match="Groq only supports image files"),
            ):
                provider.chat_completion_with_files(
                    [{"role": "user", "content": "see"}], files=["doc.pdf"]
                )

    def test_gemini_with_files(self):
        """Test: Gemini provider with files."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Gemini vision"

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = GeminiProvider(llm="gemini-1.5-pro", api_key="AIzaSy-test")

            with patch(
                "llm_client.utils.file_utils.prepare_files_for_provider",
                return_value=[{"type": "image_url", "image_url": {"url": "..."}}],
            ):
                response = provider.chat_completion_with_files(
                    [{"role": "user", "content": "see"}], files=["img.jpg"]
                )
                assert response == "Gemini vision"

    def test_ollama_with_invalid_file_type_raises_error(self):
        """Test: Ollama raises error for non-image files."""
        with patch("llm_client.providers.providers.Client", MagicMock()):
            provider = OllamaProvider(llm="llava")

            with (
                patch("llm_client.utils.file_utils.detect_file_type", return_value="pdf"),
                pytest.raises(
                    ChatCompletionError, match="Ollama vision models only support image files"
                ),
            ):
                provider.chat_completion_with_files(
                    [{"role": "user", "content": "see"}], files=["doc.pdf"]
                )

    def test_ollama_tools_not_implemented(self):
        """Test: Ollama tools raise NotImplementedError."""
        with patch("llm_client.providers.providers.Client", MagicMock()):
            provider = OllamaProvider(llm="llama3")
            with pytest.raises(
                NotImplementedError, match="OllamaProvider does not support tool calling"
            ):
                provider.chat_completion_with_tools([], [])

    def test_openai_none_content(self):
        """Test: OpenAI handles None content."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = None

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm="gpt-4o", api_key="sk-test")
            response = provider.chat_completion([])
            assert response is None

    def test_openai_stream_no_client(self):
        """Test: OpenAI stream raises error if no client."""
        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            provider = OpenAIProvider(llm="gpt-4o", api_key="sk-test")
            provider.client = None
            # Since _chat_completion_stream_impl is a generator, the error
            # is raised during iteration, not when calling the method.
            # And it's not wrapped by BaseProvider because it's already returned.
            with pytest.raises(RuntimeError, match="OpenAI client not initialized"):
                list(provider.chat_completion_stream([]))

    def test_groq_with_files_no_user_msg(self):
        """Test: Groq with files when no user message exists."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Groq vision"

        with patch("llm_client.providers.providers.Groq") as mock_groq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_groq.return_value = mock_client

            provider = GroqProvider(llm="llama-3.2-11b-vision-preview", api_key="gsk-test")

            with (
                patch("llm_client.utils.file_utils.detect_file_type", return_value="image"),
                patch(
                    "llm_client.utils.file_utils.prepare_files_for_provider",
                    return_value=[{"type": "image_url", "image_url": {"url": "..."}}],
                ),
            ):
                # No messages provided
                response = provider.chat_completion_with_files([], files=["img.jpg"])
                assert response == "Groq vision"

    def test_gemini_with_files_no_user_msg(self):
        """Test: Gemini with files when no user message exists."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Gemini vision"

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = GeminiProvider(llm="gemini-1.5-pro", api_key="AIzaSy-test")

            with patch(
                "llm_client.utils.file_utils.prepare_files_for_provider",
                return_value=[{"type": "image_url", "image_url": {"url": "..."}}],
            ):
                response = provider.chat_completion_with_files([], files=["img.jpg"])
                assert response == "Gemini vision"

    def test_all_providers_no_client_errors(self):
        """Test: All providers raise error when client is not initialized."""
        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            openai = OpenAIProvider(llm="gpt-4o", api_key="sk-test")
            openai.client = None
            with pytest.raises(ChatCompletionError):
                openai.chat_completion_with_files([], [])
            with pytest.raises(ChatCompletionError):
                openai.chat_completion_with_tools([], [])

        with patch("llm_client.providers.providers.Groq", MagicMock()):
            groq = GroqProvider(llm="llama3", api_key="gsk-test")
            groq.client = None
            with pytest.raises(ChatCompletionError):
                groq.chat_completion([])
            with pytest.raises(ChatCompletionError):
                groq.chat_completion_with_files([], [])
            with pytest.raises(ChatCompletionError):
                groq.chat_completion_with_tools([], [])
            with pytest.raises(RuntimeError):
                list(groq.chat_completion_stream([]))

        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            gemini = GeminiProvider(llm="gemini", api_key="AIzaSy")
            gemini.client = None
            with pytest.raises(ChatCompletionError):
                gemini.chat_completion([])
            with pytest.raises(ChatCompletionError):
                gemini.chat_completion_with_files([], [])
            with pytest.raises(ChatCompletionError):
                gemini.chat_completion_with_tools([], [])
            with pytest.raises(RuntimeError):
                list(gemini.chat_completion_stream([]))

    def test_all_providers_none_content(self):
        """Test: Providers handle None content."""
        # Groq
        mock_response = MagicMock()
        mock_response.choices[0].message.content = None
        with patch("llm_client.providers.providers.Groq") as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = mock_response
            groq = GroqProvider(llm="llama3", api_key="gsk-test")
            assert groq.chat_completion([]) is None

        # Gemini
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            gemini = GeminiProvider(llm="gemini", api_key="AIzaSy")
            assert gemini.chat_completion([]) is None

    def test_ollama_none_content(self):
        """Test: Ollama handles None content (though unlikely from API)."""
        mock_response = {"message": {"content": None}}
        with patch("llm_client.providers.providers.Client") as mock_client:
            mock_client.return_value.chat.return_value = mock_response
            provider = OllamaProvider(llm="llama3")
            # This might actually raise TypeError in the current implementation
            # if it tries len(None). Let's see.
            with pytest.raises(ChatCompletionError):
                provider.chat_completion([])

    def test_additional_error_logging_branches(self):
        """Test: Additional error branches for logging."""
        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            openai = OpenAIProvider(llm="gpt-4o", api_key="sk-test")
            openai.client = None
            # Covered by test_all_providers_no_client_errors but making sure
            with pytest.raises(ChatCompletionError):
                openai.chat_completion_with_files([], [])
            with pytest.raises(ChatCompletionError):
                openai.chat_completion_with_tools([], [])

        with patch("llm_client.providers.providers.Groq", MagicMock()):
            groq = GroqProvider(llm="llama3", api_key="gsk-test")
            groq.client = None
            with pytest.raises(ChatCompletionError):
                groq.chat_completion_with_files([], [])

        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            gemini = GeminiProvider(llm="gemini", api_key="AIzaSy")
            gemini.client = None
            with pytest.raises(ChatCompletionError):
                gemini.chat_completion_with_files([], [])

    def test_openai_with_files_new_message(self):
        """Test: OpenAI provider adds new message if last message is not user."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Response"

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm="gpt-4o", api_key="sk-test")

            with patch("llm_client.utils.file_utils.prepare_files_for_provider") as mock_prepare:
                mock_prepare.return_value = [{"type": "image_url", "image_url": {"url": "..."}}]

                # Last message is assistant
                messages = [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
                provider.chat_completion_with_files(messages, files=["test.jpg"])

                call_args = mock_client.chat.completions.create.call_args[1]
                assert len(call_args["messages"]) == 3
                assert call_args["messages"][2]["role"] == "user"


class TestProvidersCoverageExpansion:
    """Extra tests designed to cover remaining untested lines in providers.py."""

    def test_module_level_import_fallbacks(self):
        """Test import error fallback paths at module level when dependencies are missing."""
        import importlib
        import sys

        from llm_client.providers import providers

        with patch.dict(sys.modules, {"openai": None, "groq": None, "ollama": None}):
            importlib.reload(providers)
            assert providers.OpenAI is None
            assert providers.Groq is None
            assert providers.Client is None
            assert providers.OLLAMA_AVAILABLE is False

        # Restore original state
        importlib.reload(providers)

    def test_list_models_uninitialized_checks(self):
        """Test list_models returns [] when client is not initialized."""
        # 1. OpenAI (line 264)
        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            p_openai = OpenAIProvider(llm="gpt-4o", api_key="sk-test")
            p_openai.client = None
            assert p_openai.list_models() == []

        # 2. Groq (line 580)
        with patch("llm_client.providers.providers.Groq", MagicMock()):
            p_groq = GroqProvider(llm="llama-3.3-70b-versatile", api_key="gsk-test")
            p_groq.client = None
            assert p_groq.list_models() == []

        # 3. Gemini (line 805)
        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            p_gemini = GeminiProvider(llm="gemini-3.1-flash-lite", api_key="AIzaSy-test")
            p_gemini.client = None
            assert p_gemini.list_models() == []

        # 4. Ollama (line 1112)
        with patch("llm_client.providers.providers.Client", MagicMock()):
            p_ollama = OllamaProvider(llm="llama3.2")
            p_ollama.client = None
            assert p_ollama.list_models() == []

        # 5. KI Connect (line 1246)
        from llm_client.providers.providers import KIConnectProvider

        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            p_kic = KIConnectProvider(llm="openai-gpt5.5", api_key="key")
            p_kic.client = None
            assert p_kic.list_models() == []

    def test_groq_sync_fallback(self):
        """Test Groq fallback retry logic in sync chat completion when rate limit is exceeded (lines 328-335)."""
        from groq import APIStatusError

        provider = GroqProvider(llm="qwen/qwen3-32b", api_key="gsk-test")
        mock_client = MagicMock()
        provider.client = mock_client

        error_response = MagicMock()
        error_response.status_code = 413
        error_message = (
            "Rate limit exceeded on tokens per minute (TPM): Limit 10000, Requested 21142"
        )

        error = APIStatusError(
            message=error_message,
            response=error_response,
            body={
                "error": {"message": error_message, "type": "tokens", "code": "rate_limit_exceeded"}
            },
        )
        error.__str__ = lambda self: error_message

        mock_response_success = MagicMock()
        mock_response_success.choices[0].message.content = "Success with fallback"

        mock_client.chat.completions.create.side_effect = [error, mock_response_success]

        messages = [{"role": "user", "content": "Large request"}]

        with patch.object(
            GroqProvider,
            "_find_fallback_model",
            return_value="meta-llama/llama-4-scout-17b-16e-instruct",
        ):
            res = provider.chat_completion(messages)
            assert res == "Success with fallback"
            assert provider.llm == "meta-llama/llama-4-scout-17b-16e-instruct"

    def test_groq_validation_and_parsing_fallbacks(self):
        """Test Groq fallback model finder edge cases and parsing branches."""
        from unittest.mock import mock_open

        with patch("llm_client.providers.providers.Groq", MagicMock()):
            provider = GroqProvider(llm="qwen/qwen3-32b", api_key="gsk-test")

            # 1. Exception propagation in _chat_completion_impl (line 336)
            provider.client = MagicMock()
            provider.client.chat.completions.create.side_effect = ValueError("Some generic error")
            with pytest.raises(ValueError, match="Some generic error"):
                provider._chat_completion_impl([])

            # 2. Token count parsing fails in _find_fallback_model (lines 368-369)
            assert (
                provider._find_fallback_model("Rate limit exceeded without requested count") is None
            )

            # 3. Rate limits file not found (lines 377-378)
            with patch("pathlib.Path.exists", return_value=False):
                assert provider._find_fallback_model("Requested 21142") is None

            # 4. TPM with raw integer representation and compound ignore (lines 400, 408, 410)
            mock_md_content = """# Groq Free Plan Rate Limits
| MODEL ID | RPM | RPD | TPM | TPD | ASH | ASD |
| --- | --- | --- | --- | --- | --- | --- |
| groq/compound | 30 | 1K | 6,000 | 500K | - | - |
| test-model-k | 30 | 1K | 6K | 500K | - | - |
| test-model | 30 | 1K | 6,000 | 500K | - | - |
"""
            with patch("builtins.open", mock_open(read_data=mock_md_content)):
                # Request 5000 tokens, should find test-model-k or test-model with 6,000 TPM
                found = provider._find_fallback_model("Requested 5000")
                assert found in ["test-model-k", "test-model"]

            # 5. Error parsing rate limits file on exception (lines 419-421)
            with patch("builtins.open", side_effect=PermissionError("no read access")):
                assert provider._find_fallback_model("Requested 21142") is None

    def test_ollama_provider_coverage_expansion(self):
        """Test remaining lines in OllamaProvider."""
        with patch("llm_client.providers.providers.Client") as mock_client:
            # 1. Ollama Cloud APIKeyNotFoundError (lines 882-883)
            with pytest.raises(APIKeyNotFoundError):
                OllamaProvider(llm="model-cloud", api_key=None)

            # 1b. Initialize local client with custom host (lines 894-895)
            p_custom_host = OllamaProvider(llm="llama3.2", host="http://localhost:11434")
            assert p_custom_host.host == "http://localhost:11434"
            mock_client.assert_called_with(host="http://localhost:11434")

            # 2. Chat completion client unavailable error (line 959)
            provider = OllamaProvider(llm="llama3.2")
            with (
                patch("llm_client.providers.providers.OLLAMA_AVAILABLE", False),
                pytest.raises(ChatCompletionError, match="ollama provider not available"),
            ):
                provider.chat_completion([])

            # 3. Chat completion stream client unavailable error (line 988)
            with (
                patch("llm_client.providers.providers.OLLAMA_AVAILABLE", False),
                pytest.raises(ProviderNotAvailableError, match="ollama provider not available"),
            ):
                list(provider.chat_completion_stream([]))

            # 4. File uploads with empty enhanced_messages list (line 1013)
            provider_vision = OllamaProvider(llm="llava")
            with (
                patch("llm_client.utils.file_utils.detect_file_type", return_value="image"),
                patch("llm_client.utils.file_utils.encode_file_base64", return_value="base64"),
            ):
                provider_vision.chat_completion_with_files([], files=["image.png"])
                # Verifies that it constructs correctly and executes without error
                assert provider_vision.client.chat.called

            # 4b. Chat completion with files when client unavailable (line 988 inside _chat_completion_with_files_impl)
            with (
                patch("llm_client.providers.providers.OLLAMA_AVAILABLE", False),
                pytest.raises(ChatCompletionError, match="ollama provider not available"),
            ):
                provider_vision.chat_completion_with_files([], files=["image.png"])

            # 5. Tool calling client unavailable error (line 1051)
            provider_tools = OllamaProvider(llm="llama3.2")
            with (
                patch("llm_client.providers.providers.OLLAMA_AVAILABLE", False),
                pytest.raises(ChatCompletionError, match="ollama provider not available"),
            ):
                provider_tools.chat_completion_with_tools([], [])

            # 6. repr method (lines 1122-1123)
            p_local = OllamaProvider(llm="llama3")
            assert "mode=local" in repr(p_local)

            p_cloud = OllamaProvider(llm="llama3-cloud", api_key="sk")
            assert "mode=cloud" in repr(p_cloud)

    def test_kiconnect_provider_coverage_expansion(self):
        """Test remaining lines in KIConnectProvider."""
        from llm_client.providers.providers import KIConnectProvider

        # 0. Test KIConnect get_default_model (line 1232)
        assert KIConnectProvider.get_default_model() == "openai-gpt5.5"

        # 1. ProviderNotAvailableError when OpenAI not installed (lines 1144-1145)
        with (
            patch("llm_client.providers.providers.OpenAI", None),
            pytest.raises(ProviderNotAvailableError, match="kiconnect provider not available"),
        ):
            KIConnectProvider(llm="openai-gpt5.5", api_key="key")

        # 1b. APIKeyNotFoundError when key is missing (lines 1149-1150)
        with (
            patch("llm_client.providers.providers.OpenAI", MagicMock()),
            pytest.raises(APIKeyNotFoundError, match="KICONNECT_API_KEY not found"),
        ):
            KIConnectProvider(llm="openai-gpt5.5", api_key=None)

        # 2. RuntimeError when client is uninitialized in _chat_completion_impl (line 1172)
        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            p_kic = KIConnectProvider(llm="openai-gpt5.5", api_key="key")
            p_kic.client = None
            with pytest.raises(ChatCompletionError, match="KI Connect client not initialized"):
                p_kic.chat_completion([])

        # 3. extra_content branch (line 1189) and None response content (lines 1192-1193)
        with patch("llm_client.providers.providers.OpenAI"):
            mock_client = MagicMock()
            p_kic = KIConnectProvider(llm="openai-gpt5.5", api_key="key")
            p_kic.client = mock_client

            # With extra_content (line 1189)
            mock_message = MagicMock()
            mock_message.content = "resp content"
            mock_message.extra_content = "some reasoning"
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=mock_message)]
            mock_client.chat.completions.create.return_value = mock_response

            assert p_kic.chat_completion([]) == "resp content"

            # With None content (lines 1192-1193)
            mock_message.content = None
            assert p_kic.chat_completion([]) is None

        # 4. Stream client not initialized RuntimeError (line 1210)
        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            p_kic = KIConnectProvider(llm="openai-gpt5.5", api_key="key")
            p_kic.client = None
            with pytest.raises(RuntimeError, match="KI Connect client not initialized"):
                list(p_kic.chat_completion_stream([]))

        # 5. Stream chunk yielding (line 1223)
        with patch("llm_client.providers.providers.OpenAI"):
            mock_client = MagicMock()
            p_kic = KIConnectProvider(llm="openai-gpt5.5", api_key="key")
            p_kic.client = mock_client

            mock_chunk_1 = MagicMock()
            mock_chunk_1.choices = [MagicMock(delta=MagicMock(content="token1"))]
            mock_chunk_2 = MagicMock()
            mock_chunk_2.choices = [MagicMock(delta=MagicMock(content="token2"))]

            mock_client.chat.completions.create.return_value = [mock_chunk_1, mock_chunk_2]

            chunks = list(p_kic.chat_completion_stream([]))
            assert chunks == ["token1", "token2"]

        # 6. list_models raises Exception (lines 1255-1257)
        with patch("llm_client.providers.providers.OpenAI", MagicMock()):
            p_kic = KIConnectProvider(llm="openai-gpt5.5", api_key="key")
            with patch("requests.get", side_effect=Exception("Timeout")):
                assert p_kic.list_models() == []

            # 6b. list_models success (line 1255)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"data": [{"id": "model-kic-1"}]}
            with patch("requests.get", return_value=mock_resp):
                assert p_kic.list_models() == ["model-kic-1"]
