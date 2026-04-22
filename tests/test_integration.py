"""Integration tests for llm_client package.

These tests verify that all components work together correctly.
"""

from unittest.mock import MagicMock, patch

import pytest

from llm_client import LLMClient, ProviderFactory
from llm_client.exceptions import (
    APIKeyNotFoundError,
    InvalidProviderError,
    ProviderNotAvailableError,
)
from llm_client.providers.providers import (
    GeminiProvider,
    GroqProvider,
    OllamaProvider,
    OpenAIProvider,
)


class TestEndToEndWorkflow:
    """End-to-end tests for complete workflows."""

    def test_openai_end_to_end(self, monkeypatch):
        """Test: Complete workflow with OpenAI provider."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello! How can I help?"

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            # Create client
            client = LLMClient(api_choice="openai", llm="gpt-4o", temperature=0.5)

            # Verify initialization
            assert client.api_choice == "openai"
            assert client.llm == "gpt-4o"
            assert client.temperature == 0.5

            # Perform chat completion
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ]
            response = client.chat_completion(messages)

            assert response == "Hello! How can I help?"

    def test_groq_end_to_end(self, monkeypatch):
        """Test: Complete workflow with Groq provider."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Groq response here"

        with patch("llm_client.providers.providers.Groq") as mock_groq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_groq.return_value = mock_client

            client = LLMClient(api_choice="groq")
            messages = [{"role": "user", "content": "Test"}]
            response = client.chat_completion(messages)

            assert response == "Groq response here"

    def test_gemini_end_to_end(self, monkeypatch):
        """Test: Complete workflow with Gemini provider."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Gemini response"

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="gemini", llm="gemini-2.5-flash")
            messages = [{"role": "user", "content": "Explain quantum"}]
            response = client.chat_completion(messages)

            assert response == "Gemini response"

    def test_ollama_end_to_end(self):
        """Test: Complete workflow with Ollama provider."""
        mock_response = {"message": {"content": "Ollama local response"}}

        with patch("llm_client.providers.providers.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.chat.return_value = mock_response
            mock_client.return_value = mock_instance

            client = LLMClient(api_choice="ollama", llm="llama3.2:1b")
            messages = [{"role": "user", "content": "Hello"}]
            response = client.chat_completion(messages)

            assert response == "Ollama local response"


class TestProviderSwitching:
    """Integration tests for provider switching."""

    def test_switch_openai_to_gemini_full_workflow(self, monkeypatch):
        """Test: Full workflow switching from OpenAI to Gemini."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")

        openai_response = MagicMock()
        openai_response.choices[0].message.content = "OpenAI says hello"

        gemini_response = MagicMock()
        gemini_response.choices[0].message.content = "Gemini says hello"

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Start with OpenAI
            client = LLMClient(api_choice="openai")
            mock_client.chat.completions.create.return_value = openai_response

            messages = [{"role": "user", "content": "Hello"}]
            response1 = client.chat_completion(messages)
            assert response1 == "OpenAI says hello"
            assert isinstance(client.provider, OpenAIProvider)

            # Switch to Gemini
            client.switch_provider("gemini")
            mock_client.chat.completions.create.return_value = gemini_response

            response2 = client.chat_completion(messages)
            assert response2 == "Gemini says hello"
            assert isinstance(client.provider, GeminiProvider)

    def test_switch_with_parameter_updates(self, monkeypatch):
        """Test: Switching providers with parameter updates."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        with (
            patch("llm_client.providers.providers.OpenAI") as mock_openai,
            patch("llm_client.providers.providers.Groq") as mock_groq,
        ):
            mock_openai.return_value = MagicMock()
            mock_groq.return_value = MagicMock()

            client = LLMClient(api_choice="openai", llm="gpt-4o", temperature=0.7, max_tokens=512)

            assert client.llm == "gpt-4o"
            assert client.temperature == 0.7
            assert client.max_tokens == 512

            # Switch with new parameters
            client.switch_provider(
                "groq", llm="llama-3.3-70b-versatile", temperature=0.3, max_tokens=1024
            )

            assert client.llm == "llama-3.3-70b-versatile"
            assert client.temperature == 0.3
            assert client.max_tokens == 1024


class TestFactoryAndClientIntegration:
    """Tests for integration between Factory and Client."""

    def test_client_uses_factory_correctly(self, monkeypatch):
        """Test: LLMClient uses ProviderFactory correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai")

            # Client should have created provider via factory
            assert isinstance(client.provider, OpenAIProvider)

    def test_factory_default_models_match_providers(self):
        """Test: Factory uses correct default models from providers."""
        with (
            patch("llm_client.providers.providers.OpenAI") as mock_openai,
            patch("llm_client.providers.providers.Groq") as mock_groq,
            patch("llm_client.providers.providers.Client") as mock_client,
        ):
            mock_openai.return_value = MagicMock()
            mock_groq.return_value = MagicMock()
            mock_ollama_instance = MagicMock()
            mock_client.return_value = mock_ollama_instance

            # Test each provider's default model
            openai_provider = ProviderFactory.create_provider(
                api_choice="openai", llm=None, openai_api_key="sk-test"
            )
            assert openai_provider.llm == "gpt-4o-mini"

            groq_provider = ProviderFactory.create_provider(
                api_choice="groq", llm=None, groq_api_key="gsk-test"
            )
            assert groq_provider.llm == "qwen/qwen3-32b"

            gemini_provider = ProviderFactory.create_provider(
                api_choice="gemini", llm=None, gemini_api_key="AIzaSy-test"
            )
            assert gemini_provider.llm == "gemini-2.0-flash-exp"

            ollama_provider = ProviderFactory.create_provider(api_choice="ollama", llm=None)
            assert ollama_provider.llm == "llama3.2:1b"


class TestMultiProviderScenarios:
    """Tests for scenarios using multiple providers."""

    def test_parallel_clients_different_providers(self, monkeypatch):
        """Test: Multiple clients with different providers work independently."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        openai_response = MagicMock()
        openai_response.choices[0].message.content = "OpenAI"

        groq_response = MagicMock()
        groq_response.choices[0].message.content = "Groq"

        with (
            patch("llm_client.providers.providers.OpenAI") as mock_openai,
            patch("llm_client.providers.providers.Groq") as mock_groq,
        ):
            openai_client_mock = MagicMock()
            openai_client_mock.chat.completions.create.return_value = openai_response
            mock_openai.return_value = openai_client_mock

            groq_client_mock = MagicMock()
            groq_client_mock.chat.completions.create.return_value = groq_response
            mock_groq.return_value = groq_client_mock

            client1 = LLMClient(api_choice="openai")
            client2 = LLMClient(api_choice="groq")

            messages = [{"role": "user", "content": "Test"}]

            response1 = client1.chat_completion(messages)
            response2 = client2.chat_completion(messages)

            assert response1 == "OpenAI"
            assert response2 == "Groq"
            assert client1.api_choice != client2.api_choice

    def test_sequential_provider_usage(self, monkeypatch):
        """Test: Using different providers sequentially."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")

        with (
            patch("llm_client.providers.providers.OpenAI") as mock_openai,
            patch("llm_client.providers.providers.Groq") as mock_groq,
        ):
            mock_openai.return_value = MagicMock()
            mock_groq.return_value = MagicMock()

            # messages = [{"role": "user", "content": "Test"}]

            # Use OpenAI
            client = LLMClient(api_choice="openai")
            assert isinstance(client.provider, OpenAIProvider)

            # Switch to Groq
            client.switch_provider("groq")
            assert isinstance(client.provider, GroqProvider)

            # Switch to Gemini
            client.switch_provider("gemini")
            assert isinstance(client.provider, GeminiProvider)

            # Switch to Ollama
            client.switch_provider("ollama")
            assert isinstance(client.provider, OllamaProvider)


class TestErrorHandlingIntegration:
    """Integration tests for error handling across components."""

    def test_missing_api_key_propagates_correctly(self, monkeypatch):
        """Test: Missing API key error propagates from provider through client."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(
            APIKeyNotFoundError,
            match="OPENAI_API_KEY not found for openai provider. Please set it in environment or pass explicitly.",
        ):
            LLMClient(api_choice="openai")

    def test_invalid_provider_switch_propagates(self, monkeypatch):
        """Test: Invalid provider switch error propagates correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai")

            with pytest.raises(
                InvalidProviderError,
                match="Invalid provider: nonexistent. Valid providers are: openai, groq, gemini, ollama",
            ):
                client.switch_provider("nonexistent")

    def test_package_not_available_error(self, monkeypatch):
        """Test: Package not available error is handled correctly."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        with (
            patch("llm_client.providers.providers.Groq", None),
            pytest.raises(
                ProviderNotAvailableError,
                match="groq provider not available. Install with: pip install groq",
            ),
        ):
            LLMClient(api_choice="groq")


class TestRealWorldScenarios:
    """Tests simulating real-world usage scenarios."""

    def test_fallback_strategy(self, monkeypatch):
        """Test: Fallback strategy when primary provider fails."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        with (
            patch("llm_client.providers.providers.OpenAI") as mock_openai,
            patch("llm_client.providers.providers.Groq") as mock_groq,
        ):
            mock_openai_client = MagicMock()
            mock_groq_client = MagicMock()

            mock_openai.return_value = mock_openai_client
            mock_groq.return_value = mock_groq_client

            # Simulate OpenAI failure
            mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

            groq_response = MagicMock()
            groq_response.choices[0].message.content = "Groq fallback"
            mock_groq_client.chat.completions.create.return_value = groq_response

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Test"}]

            # Try OpenAI, expect failure
            with pytest.raises(Exception, match="API Error"):
                client.chat_completion(messages)

            # Fallback to Groq
            client.switch_provider("groq")
            response = client.chat_completion(messages)

            assert response == "Groq fallback"

    def test_cost_optimization_workflow(self, monkeypatch):
        """Test: Workflow switching between providers for cost optimization."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        with (
            patch("llm_client.providers.providers.OpenAI") as mock_openai,
            patch("llm_client.providers.providers.Groq") as mock_groq,
        ):
            openai_mock = MagicMock()
            groq_mock = MagicMock()

            mock_openai.return_value = openai_mock
            mock_groq.return_value = groq_mock

            simple_response = MagicMock()
            simple_response.choices[0].message.content = "Simple answer"

            complex_response = MagicMock()
            complex_response.choices[0].message.content = "Complex answer"

            groq_mock.chat.completions.create.return_value = simple_response
            openai_mock.chat.completions.create.return_value = complex_response

            client = LLMClient(api_choice="groq")

            # Simple task with cheaper model
            simple_messages = [{"role": "user", "content": "What is 2+2?"}]
            response1 = client.chat_completion(simple_messages)
            assert response1 == "Simple answer"

            # Complex task with more capable model
            client.switch_provider("openai", llm="gpt-4o")
            complex_messages = [{"role": "user", "content": "Explain quantum computing"}]
            response2 = client.chat_completion(complex_messages)
            assert response2 == "Complex answer"

    def test_conversation_across_providers(self, monkeypatch):
        """Test: Maintaining conversation context across provider switches."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        with (
            patch("llm_client.providers.providers.OpenAI") as mock_openai,
            patch("llm_client.providers.providers.Groq") as mock_groq,
        ):
            mock_openai.return_value = MagicMock()
            mock_groq.return_value = MagicMock()

            client = LLMClient(api_choice="openai")

            # Build conversation history
            conversation = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "Tell me about AI"},
            ]

            # Switch provider but maintain conversation
            client.switch_provider("groq")

            # Conversation history should still work
            assert len(conversation) == 3
            assert conversation[0]["role"] == "user"


class TestProviderPropertiesIntegration:
    """Integration tests for provider properties."""

    def test_llm_property_consistency(self, monkeypatch):
        """Test: llm property is consistent across client and provider."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai", llm="gpt-4o")

            assert client.llm == "gpt-4o"
            assert client.provider.llm == "gpt-4o"
            assert client.llm == client.provider.llm

    def test_temperature_consistency_after_switch(self, monkeypatch):
        """Test: Temperature is maintained correctly during switches."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        with (
            patch("llm_client.providers.providers.OpenAI") as mock_openai,
            patch("llm_client.providers.providers.Groq") as mock_groq,
        ):
            mock_openai.return_value = MagicMock()
            mock_groq.return_value = MagicMock()

            client = LLMClient(api_choice="openai", temperature=0.5)
            assert client.temperature == 0.5

            # Switch without specifying temperature
            client.switch_provider("groq")
            assert client.temperature == 0.5  # Should be preserved

            # Switch with new temperature
            client.switch_provider("openai", temperature=0.8)
            assert client.temperature == 0.8

    def test_client_property_backward_compatibility(self, monkeypatch):
        """Test: client property provides backward compatibility."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")

            # client property should return provider's client
            assert client.client == client.provider.client
            assert client.client == mock_client
