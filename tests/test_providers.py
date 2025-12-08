"""Unit tests for provider implementations."""

from unittest.mock import MagicMock, patch

import pytest

from llm_client.exceptions import (
    APIKeyNotFoundError,
    ChatCompletionError,
    ProviderNotAvailableError,
)
from llm_client.providers import (
    GeminiProvider,
    GroqProvider,
    OllamaProvider,
    OpenAIProvider,
)


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_initialization_success(self):
        """Test: OpenAI provider initializes correctly with valid API key."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
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
            patch("llm_client.providers.OpenAI", None),
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

        with patch("llm_client.providers.OpenAI") as mock_openai:
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
        with patch("llm_client.providers.OpenAI") as mock_openai:
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
        with patch("llm_client.providers.OpenAI", MagicMock()):
            assert OpenAIProvider.is_available() is True

    def test_is_available_when_not_installed(self):
        """Test: is_available returns False when package is not installed."""
        with patch("llm_client.providers.OpenAI", None):
            assert OpenAIProvider.is_available() is False

    def test_repr(self):
        """Test: __repr__ returns correct string representation."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
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
        with patch("llm_client.providers.OpenAI") as mock_openai:
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
        with patch("llm_client.providers.Groq") as mock_groq:
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
            patch("llm_client.providers.Groq", None),
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

        with patch("llm_client.providers.Groq") as mock_groq:
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
        with patch("llm_client.providers.Groq") as mock_groq:
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
        assert GroqProvider.get_default_model() == "moonshotai/kimi-k2-instruct-0905"

    def test_is_available_when_installed(self):
        """Test: is_available returns True when package is installed."""
        with patch("llm_client.providers.Groq", MagicMock()):
            assert GroqProvider.is_available() is True

    def test_is_available_when_not_installed(self):
        """Test: is_available returns False when package is not installed."""
        with patch("llm_client.providers.Groq", None):
            assert GroqProvider.is_available() is False


class TestGeminiProvider:
    """Tests for GeminiProvider."""

    def test_initialization_success(self):
        """Test: Gemini provider initializes correctly with valid API key."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
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
            patch("llm_client.providers.OpenAI", None),
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

        with patch("llm_client.providers.OpenAI") as mock_openai:
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
        with patch("llm_client.providers.OpenAI") as mock_openai:
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
        assert GeminiProvider.get_default_model() == "gemini-2.0-flash-exp"

    def test_is_available_when_installed(self):
        """Test: is_available returns True when OpenAI package is installed."""
        with patch("llm_client.providers.OpenAI", MagicMock()):
            assert GeminiProvider.is_available() is True

    def test_is_available_when_not_installed(self):
        """Test: is_available returns False when OpenAI package is not installed."""
        with patch("llm_client.providers.OpenAI", None):
            assert GeminiProvider.is_available() is False

    def test_uses_correct_base_url(self):
        """Test: Gemini uses correct API base URL."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
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
        with patch("llm_client.providers.Client") as mock_client:
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
            patch("llm_client.providers.Client", None),
            patch("llm_client.providers.OLLAMA_AVAILABLE", False),
            pytest.raises(
                ProviderNotAvailableError,
                match="ollama provider not available. Install with: pip install ollama",
            ),
        ):
            OllamaProvider(llm="llama3.2:1b")

    def test_chat_completion_success(self):
        """Test: Chat completion returns correct response."""
        mock_response = {"message": {"content": "Ollama response"}}

        with patch("llm_client.providers.Client") as mock_client:
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
        with patch("llm_client.providers.Client", MagicMock()):
            provider = OllamaProvider(llm="llama3.2:1b")

        with (
            patch("llm_client.providers.Client", None),
            patch("llm_client.providers.OLLAMA_AVAILABLE", False),
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
        with patch("llm_client.providers.Client", MagicMock()):
            assert OllamaProvider.is_available() is True

    def test_is_available_when_not_installed(self):
        """Test: is_available returns False when package is not installed."""
        with (
            patch("llm_client.providers.Client", None),
            patch("llm_client.providers.OLLAMA_AVAILABLE", False),
        ):
            assert OllamaProvider.is_available() is False

    def test_custom_keep_alive_parameter(self):
        """Test: Custom keep_alive parameter is stored correctly."""
        with patch("llm_client.providers.Client", MagicMock()):
            provider = OllamaProvider(llm="llama3.2:1b", keep_alive="15m")
            assert provider.keep_alive == "15m"

    def test_custom_options_in_chat_completion(self):
        """Test: Custom parameters are passed correctly to ollama.chat."""
        mock_response = {"message": {"content": "Response"}}

        with patch("llm_client.providers.Client") as mock_client:
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
        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            # Very low temperature
            provider = OpenAIProvider(llm="gpt-4o", temperature=0.0, api_key="sk-test")
            assert provider.temperature == 0.0

            # Very high temperature
            provider = OpenAIProvider(llm="gpt-4o", temperature=2.0, api_key="sk-test")
            assert provider.temperature == 2.0

    def test_extreme_max_tokens_values(self):
        """Test: Providers handle extreme max_tokens values."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
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

        with patch("llm_client.providers.OpenAI") as mock_openai:
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

        with patch("llm_client.providers.OpenAI") as mock_openai:
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
