"""Tests for Anthropic provider (sync and async)."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from llm_client import LLMClient, ProviderFactory, AnthropicProvider
from llm_client.providers.async_providers import AsyncAnthropicProvider
from llm_client.exceptions import APIKeyNotFoundError, ProviderNotAvailableError


class TestAnthropicProvider:
    """Test suite for AnthropicProvider."""

    def test_get_default_model(self):
        """Test default model for Anthropic provider."""
        assert AnthropicProvider.get_default_model() == "claude-3-5-sonnet-20241022"

    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises APIKeyNotFoundError."""
        with pytest.raises(APIKeyNotFoundError):
            AnthropicProvider(llm="claude-3-5-sonnet-20241022", api_key=None)

    @patch("llm_client.providers.providers.Anthropic")
    def test_chat_completion_success(self, mock_anthropic_class):
        """Test successful chat completion."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Hello from Claude!"

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(llm="claude-3-5-sonnet-20241022", api_key="sk-ant-test")
        messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]

        response = provider.chat_completion(messages)
        assert response == "Hello from Claude!"
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hi"}],
            system="Be brief.",
            max_tokens=512,
            temperature=0.7,
        )

    @patch("llm_client.providers.providers.Anthropic")
    def test_chat_completion_stream(self, mock_anthropic_class):
        """Test streaming completion."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.text_stream = iter(["Hello", " world"])
        mock_client.messages.stream.return_value.__enter__.return_value = mock_stream

        provider = AnthropicProvider(llm="claude-3-5-sonnet-20241022", api_key="sk-ant-test")
        chunks = list(provider.chat_completion_stream([{"role": "user", "content": "Hi"}]))
        assert chunks == ["Hello", " world"]

    @patch("llm_client.providers.providers.Anthropic")
    def test_chat_completion_with_tools(self, mock_anthropic_class):
        """Test completion with tool calls."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Checking weather..."

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "toolu_123"
        mock_tool_block.name = "get_weather"
        mock_tool_block.input = {"location": "Berlin"}

        mock_response = MagicMock()
        mock_response.content = [mock_text_block, mock_tool_block]
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(llm="claude-3-5-sonnet-20241022", api_key="sk-ant-test")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                },
            }
        ]

        result = provider.chat_completion_with_tools([{"role": "user", "content": "Weather in Berlin?"}], tools)
        assert result["content"] == "Checking weather..."
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        assert "Berlin" in result["tool_calls"][0]["function"]["arguments"]

    @patch("llm_client.providers.providers.Anthropic")
    def test_list_models(self, mock_anthropic_class):
        """Test list_models method."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_model = MagicMock()
        mock_model.id = "claude-3-5-sonnet-20241022"
        mock_client.models.list.return_value.data = [mock_model]

        provider = AnthropicProvider(llm="claude-3-5-sonnet-20241022", api_key="sk-ant-test")
        models = provider.list_models()
        assert "claude-3-5-sonnet-20241022" in models


class TestAsyncAnthropicProvider:
    """Test suite for AsyncAnthropicProvider."""

    @pytest.mark.asyncio
    @patch("llm_client.providers.async_providers.AsyncAnthropic")
    async def test_achat_completion_success(self, mock_async_anthropic):
        """Test async chat completion."""
        mock_client = MagicMock()
        mock_async_anthropic.return_value = mock_client

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Async response from Claude!"

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AsyncAnthropicProvider(llm="claude-3-5-sonnet-20241022", api_key="sk-ant-test")
        response = await provider.achat_completion([{"role": "user", "content": "Hello"}])
        assert response == "Async response from Claude!"

    @pytest.mark.asyncio
    @patch("llm_client.providers.async_providers.AsyncAnthropic")
    async def test_achat_completion_stream(self, mock_async_anthropic):
        """Test async streaming completion."""
        mock_client = MagicMock()
        mock_async_anthropic.return_value = mock_client

        class AsyncStreamMock:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                pass

            async def __aiter__(self):
                for chunk in ["Async", " stream"]:
                    yield chunk

            @property
            def text_stream(self):
                return self

        mock_client.messages.stream.return_value = AsyncStreamMock()

        provider = AsyncAnthropicProvider(llm="claude-3-5-sonnet-20241022", api_key="sk-ant-test")
        chunks = []
        async for chunk in provider.achat_completion_stream([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        assert chunks == ["Async", " stream"]
