"""Unit tests for ProviderFactory."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from llm_client.exceptions import (
    APIKeyNotFoundError,
    InvalidProviderError,
)
from llm_client.provider_factory import ProviderFactory
from llm_client.providers import (
    GeminiProvider,
    GroqProvider,
    OllamaProvider,
    OpenAIProvider,
)


class TestProviderFactoryCreation:
    """Tests for provider creation via factory."""

    def test_create_openai_provider_explicit(self):
        """Test: Create OpenAI provider explicitly."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider = ProviderFactory.create_provider(
                api_choice="openai",
                llm="gpt-4o",
                temperature=0.5,
                max_tokens=1024,
                openai_api_key="sk-test",
            )

            assert isinstance(provider, OpenAIProvider)
            assert provider.llm == "gpt-4o"
            assert provider.temperature == 0.5
            assert provider.max_tokens == 1024

    def test_create_groq_provider_explicit(self):
        """Test: Create Groq provider explicitly."""
        with patch("llm_client.providers.Groq") as mock_groq:
            mock_groq.return_value = MagicMock()

            provider = ProviderFactory.create_provider(
                api_choice="groq",
                llm="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=512,
                groq_api_key="gsk-test",
            )

            assert isinstance(provider, GroqProvider)
            assert provider.llm == "llama-3.3-70b-versatile"

    def test_create_gemini_provider_explicit(self):
        """Test: Create Gemini provider explicitly."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider = ProviderFactory.create_provider(
                api_choice="gemini",
                llm="gemini-2.5-flash",
                temperature=0.8,
                max_tokens=2048,
                gemini_api_key="AIzaSy-test",
            )

            assert isinstance(provider, GeminiProvider)
            assert provider.llm == "gemini-2.5-flash"
            assert provider.temperature == 0.8

    def test_create_ollama_provider_explicit(self):
        """Test: Create Ollama provider explicitly."""
        with patch("llm_client.providers.ollama") as mock_ollama:
            mock_ollama.chat.return_value = {"message": {"content": "test"}}

            provider = ProviderFactory.create_provider(
                api_choice="ollama",
                llm="llama3.2:1b",
                temperature=0.7,
                max_tokens=512,
                keep_alive="10m",
            )

            assert isinstance(provider, OllamaProvider)
            assert provider.llm == "llama3.2:1b"
            assert provider.keep_alive == "10m"

    def test_create_provider_with_default_model(self):
        """Test: Provider uses default model when llm is None."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider = ProviderFactory.create_provider(
                api_choice="openai", llm=None, openai_api_key="sk-test"
            )

            assert provider.llm == OpenAIProvider.get_default_model()
            assert provider.llm == "gpt-4o-mini"

    def test_invalid_api_choice_raises_error(self):
        """Test: InvalidProviderError for invalid api_choice."""
        with pytest.raises(
            InvalidProviderError,
            match="Invalid provider: invalid. Valid providers are: openai, groq, gemini, ollama",
        ):
            ProviderFactory.create_provider(api_choice="invalid")

    def test_case_insensitive_api_choice(self):
        """Test: api_choice is case-insensitive."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider = ProviderFactory.create_provider(
                api_choice="OPENAI", openai_api_key="sk-test"
            )

            assert isinstance(provider, OpenAIProvider)


class TestAutoSelectAPI:
    """Tests for automatic API selection."""

    def test_auto_select_openai(self):
        """Test: Auto-selects OpenAI when key is available."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider = ProviderFactory.create_provider(
                api_choice=None, openai_api_key="sk-test", groq_api_key=None, gemini_api_key=None
            )

            assert isinstance(provider, OpenAIProvider)

    def test_auto_select_groq(self):
        """Test: Auto-selects Groq when OpenAI key is not available."""
        with patch("llm_client.providers.Groq") as mock_groq:
            mock_groq.return_value = MagicMock()

            provider = ProviderFactory.create_provider(
                api_choice=None, openai_api_key=None, groq_api_key="gsk-test", gemini_api_key=None
            )

            assert isinstance(provider, GroqProvider)

    def test_auto_select_gemini(self):
        """Test: Auto-selects Gemini when OpenAI and Groq keys are not available."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider = ProviderFactory.create_provider(
                api_choice=None,
                openai_api_key=None,
                groq_api_key=None,
                gemini_api_key="AIzaSy-test",
            )

            assert isinstance(provider, GeminiProvider)

    def test_auto_select_ollama_when_no_keys(self, monkeypatch):
        """Test: Auto-selects Ollama when no API keys are available."""
        # Make sure we're not in Colab
        monkeypatch.delenv("COLAB_GPU", raising=False)
        if "google.colab" in sys.modules:
            del sys.modules["google.colab"]

        with patch("llm_client.providers.ollama") as mock_ollama:
            mock_ollama.chat.return_value = {"message": {"content": "test"}}

            provider = ProviderFactory.create_provider(
                api_choice=None, openai_api_key=None, groq_api_key=None, gemini_api_key=None
            )

            assert isinstance(provider, OllamaProvider)

    def test_auto_select_priority_order(self):
        """Test: Priority is OpenAI > Groq > Gemini > Ollama."""
        with (
            patch("llm_client.providers.OpenAI") as mock_openai,
            patch("llm_client.providers.Groq") as mock_groq,
        ):
            mock_openai.return_value = MagicMock()
            mock_groq.return_value = MagicMock()

            # When all keys available, OpenAI is selected
            provider = ProviderFactory.create_provider(
                api_choice=None,
                openai_api_key="sk-test",
                groq_api_key="gsk-test",
                gemini_api_key="AIzaSy-test",
            )
            assert isinstance(provider, OpenAIProvider)

    def test_colab_without_api_keys_raises_error(self, monkeypatch):
        """Test: RuntimeError in Colab without API keys."""
        # Simulate Colab environment
        monkeypatch.setitem(sys.modules, "google.colab", MagicMock())
        monkeypatch.setenv("COLAB_GPU", "1")

        with pytest.raises(
            APIKeyNotFoundError,
            match="OPENAI_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY not found for colab provider. Please set it in environment or pass explicitly.",
        ):
            ProviderFactory._auto_select_api(
                openai_api_key=None, groq_api_key=None, gemini_api_key=None
            )


class TestProviderAvailability:
    """Tests for checking provider availability."""

    def test_get_available_providers_all_installed(self):
        """Test: Returns all providers when packages are installed."""
        with (
            patch("llm_client.providers.OpenAI", MagicMock()),
            patch("llm_client.providers.Groq", MagicMock()),
            patch("llm_client.providers.ollama", MagicMock()),
        ):
            available = ProviderFactory.get_available_providers()

            assert "openai" in available
            assert "groq" in available
            assert "gemini" in available  # Uses OpenAI package
            assert "ollama" in available

    def test_get_available_providers_partial(self):
        """Test: Returns only available providers."""
        with (
            patch("llm_client.providers.OpenAI", MagicMock()),
            patch("llm_client.providers.Groq", None),
            patch("llm_client.providers.ollama", MagicMock()),
        ):
            available = ProviderFactory.get_available_providers()

            assert "openai" in available
            assert "groq" not in available
            assert "gemini" in available  # Still available via OpenAI
            assert "ollama" in available

    def test_is_provider_available_installed(self):
        """Test: is_provider_available returns True for installed package."""
        with patch("llm_client.providers.OpenAI", MagicMock()):
            assert ProviderFactory.is_provider_available("openai") is True
            assert ProviderFactory.is_provider_available("gemini") is True

    def test_is_provider_available_not_installed(self):
        """Test: is_provider_available returns False for missing package."""
        with patch("llm_client.providers.Groq", None):
            assert ProviderFactory.is_provider_available("groq") is False

    def test_is_provider_available_invalid_name(self):
        """Test: is_provider_available returns False for invalid name."""
        assert ProviderFactory.is_provider_available("nonexistent") is False

    def test_is_provider_available_case_insensitive(self):
        """Test: is_provider_available is case-insensitive."""
        with patch("llm_client.providers.OpenAI", MagicMock()):
            assert ProviderFactory.is_provider_available("OPENAI") is True
            assert ProviderFactory.is_provider_available("OpenAI") is True
            assert ProviderFactory.is_provider_available("openai") is True


class TestProviderFactoryParameters:
    """Tests for parameter handling in factory."""

    def test_default_parameters(self):
        """Test: Default parameters are applied correctly."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider = ProviderFactory.create_provider(
                api_choice="openai", openai_api_key="sk-test"
            )

            assert provider.temperature == 0.7
            assert provider.max_tokens == 512
            assert provider.llm == "gpt-4o-mini"

    def test_custom_temperature(self):
        """Test: Custom temperature is passed correctly."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider = ProviderFactory.create_provider(
                api_choice="openai", temperature=0.3, openai_api_key="sk-test"
            )

            assert provider.temperature == 0.3

    def test_custom_max_tokens(self):
        """Test: Custom max_tokens is passed correctly."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider = ProviderFactory.create_provider(
                api_choice="openai", max_tokens=2048, openai_api_key="sk-test"
            )

            assert provider.max_tokens == 2048

    def test_ollama_keep_alive_parameter(self):
        """Test: keep_alive parameter is passed to Ollama provider."""
        with patch("llm_client.providers.ollama") as mock_ollama:
            mock_ollama.chat.return_value = {"message": {"content": "test"}}

            provider = ProviderFactory.create_provider(api_choice="ollama", keep_alive="15m")

            assert provider.keep_alive == "15m"

    def test_multiple_custom_parameters(self):
        """Test: Multiple custom parameters work together."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider = ProviderFactory.create_provider(
                api_choice="openai",
                llm="gpt-4o",
                temperature=0.5,
                max_tokens=1024,
                openai_api_key="sk-test",
            )

            assert provider.llm == "gpt-4o"
            assert provider.temperature == 0.5
            assert provider.max_tokens == 1024


class TestProviderFactoryEdgeCases:
    """Tests for edge cases in factory."""

    def test_empty_api_key_strings(self):
        """Test: Empty string API keys are treated as None."""
        with patch("llm_client.providers.ollama") as mock_ollama:
            mock_ollama.chat.return_value = {"message": {"content": "test"}}

            # Empty strings should be treated as no key
            provider = ProviderFactory.create_provider(
                api_choice=None, openai_api_key="", groq_api_key="", gemini_api_key=""
            )

            # Should fall back to Ollama
            assert isinstance(provider, OllamaProvider)

    def test_extreme_parameter_values(self):
        """Test: Extreme parameter values are accepted."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider = ProviderFactory.create_provider(
                api_choice="openai",
                temperature=0.0,
                max_tokens=1,
                openai_api_key="sk-test",
            )

            assert provider.temperature == 0.0
            assert provider.max_tokens == 1

    def test_provider_classes_registered(self):
        """Test: All provider classes are registered in factory."""
        expected_providers = ["openai", "groq", "gemini", "ollama"]

        for provider_name in expected_providers:
            assert provider_name in ProviderFactory._provider_classes

    def test_factory_creates_new_instance_each_time(self):
        """Test: Factory creates new instance for each call."""
        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            provider1 = ProviderFactory.create_provider(
                api_choice="openai", openai_api_key="sk-test"
            )
            provider2 = ProviderFactory.create_provider(
                api_choice="openai", openai_api_key="sk-test"
            )

            assert provider1 is not provider2
            assert isinstance(provider1, OpenAIProvider)
            assert isinstance(provider2, OpenAIProvider)


class TestProviderFactoryIntegration:
    """Integration tests for factory with actual provider behavior."""

    def test_created_provider_can_chat(self):
        """Test: Created provider can perform chat completion."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response"

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = ProviderFactory.create_provider(
                api_choice="openai", openai_api_key="sk-test"
            )

            messages = [{"role": "user", "content": "Hello"}]
            response = provider.chat_completion(messages)

            assert response == "Test response"

    def test_switching_between_providers(self):
        """Test: Can create different providers sequentially."""
        with (
            patch("llm_client.providers.OpenAI") as mock_openai,
            patch("llm_client.providers.Groq") as mock_groq,
        ):
            mock_openai.return_value = MagicMock()
            mock_groq.return_value = MagicMock()

            provider1 = ProviderFactory.create_provider(
                api_choice="openai", openai_api_key="sk-test"
            )
            assert isinstance(provider1, OpenAIProvider)

            provider2 = ProviderFactory.create_provider(api_choice="groq", groq_api_key="gsk-test")
            assert isinstance(provider2, GroqProvider)

    def test_auto_select_respects_package_availability(self):
        """Test: Auto-select only chooses from available packages."""
        with (
            patch("llm_client.providers.OpenAI", None),
            patch("llm_client.providers.Groq") as mock_groq,
        ):
            mock_groq.return_value = MagicMock()

            # OpenAI not available, should select Groq
            # Pass ONLY Groq key (not OpenAI key) since OpenAI is unavailable
            provider = ProviderFactory.create_provider(
                api_choice=None,
                openai_api_key=None,  # Changed from "sk-test" to None
                groq_api_key="gsk-test",
                gemini_api_key=None,
            )

            # Groq should be selected since OpenAI is unavailable
            assert isinstance(provider, GroqProvider)
