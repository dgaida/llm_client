"""Unit tests for KI Connect provider implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_client.exceptions import APIKeyNotFoundError
from llm_client.providers.async_providers import AsyncKIConnectProvider
from llm_client.providers.providers import KIConnectProvider


class TestKIConnectProvider:
    """Tests for KIConnectProvider."""

    def test_initialization_success(self):
        """Test: KI Connect provider initializes correctly with valid API key."""
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            provider = KIConnectProvider(
                llm="openai-gpt5.5", temperature=0.7, max_tokens=512, api_key="kiconnect-test"
            )

            assert provider.llm == "openai-gpt5.5"
            assert provider.client == mock_client
            mock_openai.assert_called_once_with(
                api_key="kiconnect-test", base_url="https://chat.kiconnect.nrw/api/v1"
            )

    def test_initialization_without_api_key_raises_error(self):
        """Test: APIKeyNotFoundError when API key is missing."""
        with pytest.raises(
            APIKeyNotFoundError,
            match="KICONNECT_API_KEY not found for kiconnect provider",
        ):
            KIConnectProvider(llm="openai-gpt5.5", temperature=0.7, max_tokens=512)

    def test_chat_completion_success(self):
        """Test: Chat completion returns correct response."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        # Mocking extra_content
        mock_response.choices[0].message.extra_content = None

        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = KIConnectProvider(llm="openai-gpt5.5", api_key="test")
            messages = [{"role": "user", "content": "Hello"}]
            response = provider.chat_completion(messages)

            assert response == "Test response"

    def test_get_default_model(self):
        """Test: Default model is correct."""
        assert KIConnectProvider.get_default_model() == "openai-gpt5.5"


@pytest.mark.asyncio
class TestAsyncKIConnectProvider:
    """Tests for AsyncKIConnectProvider."""

    async def test_async_initialization_success(self):
        """Test: Async KI Connect provider initializes correctly."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_client = MagicMock()
            mock_async_openai.return_value = mock_client

            provider = AsyncKIConnectProvider(llm="openai-gpt5.5", api_key="test")

            assert provider.client == mock_client
            mock_async_openai.assert_called_once_with(
                api_key="test", base_url="https://chat.kiconnect.nrw/api/v1"
            )

    async def test_achat_completion_success(self):
        """Test: Async chat completion returns correct response."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Async test response"
        mock_response.choices[0].message.extra_content = None

        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_openai.return_value = mock_client

            provider = AsyncKIConnectProvider(llm="openai-gpt5.5", api_key="test")
            messages = [{"role": "user", "content": "Hello"}]
            response = await provider.achat_completion(messages)

            assert response == "Async test response"
