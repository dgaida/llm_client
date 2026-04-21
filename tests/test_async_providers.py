"""Tests for async providers to increase coverage."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_client.exceptions import (
    APIKeyNotFoundError,
    ChatCompletionError,
    ProviderNotAvailableError,
)

# Skip all tests if async providers not available
pytestmark = pytest.mark.asyncio


class TestAsyncOpenAIProvider:
    """Tests for AsyncOpenAIProvider."""

    async def test_async_openai_initialization(self):
        """Test: AsyncOpenAIProvider initializes correctly."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_client = MagicMock()
            mock_async_openai.return_value = mock_client

            from llm_client.providers.async_providers import AsyncOpenAIProvider

            provider = AsyncOpenAIProvider(
                llm="gpt-4o", temperature=0.7, max_tokens=512, api_key="sk-test"
            )

            assert provider.llm == "gpt-4o"
            assert provider.client == mock_client

    async def test_async_openai_missing_api_key(self):
        """Test: Raises error when API key missing."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI", MagicMock()):
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            with pytest.raises(APIKeyNotFoundError):
                AsyncOpenAIProvider(llm="gpt-4o")

    async def test_async_openai_chat_completion(self):
        """Test: Async chat completion works."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Async response"

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_openai.return_value = mock_client

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")
            messages = [{"role": "user", "content": "Hello"}]

            response = await provider.achat_completion(messages)

            assert response == "Async response"

    async def test_async_openai_streaming(self):
        """Test: Async streaming works."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            async def async_generator():
                chunks = [
                    MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
                    MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))]),
                ]
                for chunk in chunks:
                    yield chunk

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=async_generator())
            mock_async_openai.return_value = mock_client

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")
            messages = [{"role": "user", "content": "Hello"}]

            result = []
            async for chunk in provider.achat_completion_stream(messages):
                result.append(chunk)

            assert result == ["Hello", " world"]

    async def test_async_openai_with_tools(self):
        """Test: Async tool calling works."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            mock_tool_call = MagicMock()
            mock_tool_call.id = "call_123"
            mock_tool_call.type = "function"
            mock_tool_call.function.name = "get_weather"
            mock_tool_call.function.arguments = '{"location": "NYC"}'

            mock_response = MagicMock()
            mock_response.choices[0].message.content = None
            mock_response.choices[0].message.tool_calls = [mock_tool_call]

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_openai.return_value = mock_client

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")
            messages = [{"role": "user", "content": "Weather in NYC?"}]
            tools = [{"type": "function", "function": {"name": "get_weather"}}]

            result = await provider.achat_completion_with_tools(messages, tools)

            assert result["content"] is None
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    async def test_sync_method_raises_runtime_error(self):
        """Test: Sync method raises RuntimeError wrapped in ChatCompletionError."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_async_openai.return_value = MagicMock()

            from llm_client.providers.async_providers import AsyncOpenAIProvider

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")

            with pytest.raises(ChatCompletionError, match="only supports async"):
                provider.chat_completion([])

    async def test_get_default_model(self):
        """Test: Get default model."""
        from llm_client.providers.async_providers import AsyncOpenAIProvider

        assert AsyncOpenAIProvider.get_default_model() == "gpt-4o-mini"

    async def test_is_available(self):
        """Test: Check if AsyncOpenAI is available."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI", MagicMock()):
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            assert AsyncOpenAIProvider.is_available() is True

        with patch("llm_client.providers.async_providers.AsyncOpenAI", None):
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            assert AsyncOpenAIProvider.is_available() is False


class TestAsyncGroqProvider:
    """Tests for AsyncGroqProvider."""

    async def test_async_groq_initialization(self):
        """Test: AsyncGroqProvider initializes correctly."""
        with patch("llm_client.providers.async_providers.AsyncGroq") as mock_async_groq:
            mock_client = MagicMock()
            mock_async_groq.return_value = mock_client

            from llm_client.providers.async_providers import AsyncGroqProvider

            provider = AsyncGroqProvider(llm="llama-3.3-70b-versatile", api_key="gsk-test")

            assert provider.llm == "llama-3.3-70b-versatile"
            assert provider.client == mock_client

    async def test_async_groq_chat_completion(self):
        """Test: Async chat completion for Groq."""
        with patch("llm_client.providers.async_providers.AsyncGroq") as mock_async_groq:
            from llm_client.providers.async_providers import AsyncGroqProvider

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Groq response"

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_groq.return_value = mock_client

            provider = AsyncGroqProvider(llm="llama-3.3-70b-versatile", api_key="gsk-test")
            messages = [{"role": "user", "content": "Hello"}]

            response = await provider.achat_completion(messages)

            assert response == "Groq response"

    async def test_async_groq_streaming(self):
        """Test: Async streaming for Groq."""
        with patch("llm_client.providers.async_providers.AsyncGroq") as mock_async_groq:
            from llm_client.providers.async_providers import AsyncGroqProvider

            async def async_generator():
                chunks = [
                    MagicMock(choices=[MagicMock(delta=MagicMock(content="Test"))]),
                    MagicMock(choices=[MagicMock(delta=MagicMock(content=" message"))]),
                ]
                for chunk in chunks:
                    yield chunk

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=async_generator())
            mock_async_groq.return_value = mock_client

            provider = AsyncGroqProvider(llm="llama-3.3-70b-versatile", api_key="gsk-test")
            messages = [{"role": "user", "content": "Hello"}]

            result = []
            async for chunk in provider.achat_completion_stream(messages):
                result.append(chunk)

            assert result == ["Test", " message"]

    async def test_async_groq_with_tools(self):
        """Test: Async tool calling for Groq."""
        with patch("llm_client.providers.async_providers.AsyncGroq") as mock_async_groq:
            from llm_client.providers.async_providers import AsyncGroqProvider

            mock_tool_call = MagicMock()
            mock_tool_call.id = "call_456"
            mock_tool_call.type = "function"
            mock_tool_call.function.name = "search"
            mock_tool_call.function.arguments = '{"query": "test"}'

            mock_response = MagicMock()
            mock_response.choices[0].message.content = None
            mock_response.choices[0].message.tool_calls = [mock_tool_call]

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_groq.return_value = mock_client

            provider = AsyncGroqProvider(llm="llama-3.3-70b-versatile", api_key="gsk-test")
            messages = [{"role": "user", "content": "Search test"}]
            tools = [{"type": "function", "function": {"name": "search"}}]

            result = await provider.achat_completion_with_tools(messages, tools)

            assert result["tool_calls"] is not None

    async def test_get_default_model_groq(self):
        """Test: Get default Groq model."""
        from llm_client.providers.async_providers import AsyncGroqProvider

        assert AsyncGroqProvider.get_default_model() == "qwen/qwen3-32b"


class TestAsyncGeminiProvider:
    """Tests for AsyncGeminiProvider."""

    async def test_async_gemini_initialization(self):
        """Test: AsyncGeminiProvider initializes correctly."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_client = MagicMock()
            mock_async_openai.return_value = mock_client

            from llm_client.providers.async_providers import AsyncGeminiProvider

            provider = AsyncGeminiProvider(llm="gemini-2.5-flash", api_key="AIzaSy-test")

            assert provider.llm == "gemini-2.5-flash"
            assert provider.client == mock_client

            # Verify correct base URL
            mock_async_openai.assert_called_once()
            call_kwargs = mock_async_openai.call_args[1]
            assert "base_url" in call_kwargs
            assert "googleapis.com" in call_kwargs["base_url"]

    async def test_async_gemini_chat_completion(self):
        """Test: Async chat completion for Gemini."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncGeminiProvider

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Gemini response"

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_openai.return_value = mock_client

            provider = AsyncGeminiProvider(llm="gemini-2.5-flash", api_key="AIzaSy-test")
            messages = [{"role": "user", "content": "Hello"}]

            response = await provider.achat_completion(messages)

            assert response == "Gemini response"

    async def test_async_gemini_streaming(self):
        """Test: Async streaming for Gemini."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncGeminiProvider

            async def async_generator():
                chunks = [
                    MagicMock(choices=[MagicMock(delta=MagicMock(content="Gemini"))]),
                    MagicMock(choices=[MagicMock(delta=MagicMock(content=" works"))]),
                ]
                for chunk in chunks:
                    yield chunk

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=async_generator())
            mock_async_openai.return_value = mock_client

            provider = AsyncGeminiProvider(llm="gemini-2.5-flash", api_key="AIzaSy-test")
            messages = [{"role": "user", "content": "Hello"}]

            result = []
            async for chunk in provider.achat_completion_stream(messages):
                result.append(chunk)

            assert result == ["Gemini", " works"]

    async def test_get_default_model_gemini(self):
        """Test: Get default Gemini model."""
        from llm_client.providers.async_providers import AsyncGeminiProvider

        assert AsyncGeminiProvider.get_default_model() == "gemini-2.0-flash-exp"


class TestAsyncProviderMixin:
    """Tests for AsyncProviderMixin."""

    async def test_achat_completion_wraps_errors(self):
        """Test: achat_completion wraps errors in ChatCompletionError."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
            mock_async_openai.return_value = mock_client

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")

            with pytest.raises(ChatCompletionError):
                await provider.achat_completion([])

    async def test_achat_completion_with_tools_not_implemented(self):
        """Test: Tool calling raises NotImplementedError if not supported."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_async_openai.return_value = MagicMock()

            from llm_client.providers.async_providers import AsyncProviderMixin

            class TestProvider(AsyncProviderMixin):
                def __init__(self):
                    self.client = None

            provider = TestProvider()

            with pytest.raises(NotImplementedError):
                await provider.achat_completion_with_tools([], [])

    async def test_streaming_wraps_errors(self):
        """Test: Streaming wraps errors in ChatCompletionError."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            async def failing_generator():
                raise Exception("Stream error")
                yield  # Never reached

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=failing_generator())
            mock_async_openai.return_value = mock_client

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")

            with pytest.raises(ChatCompletionError):
                async for _ in provider.achat_completion_stream([]):
                    pass


class TestAsyncProviderAvailability:
    """Tests for async provider availability."""

    async def test_async_openai_not_available(self):
        """Test: Raises error when AsyncOpenAI not available."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI", None):
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            with pytest.raises(ProviderNotAvailableError):
                AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")

    async def test_async_groq_not_available(self):
        """Test: Raises error when AsyncGroq not available."""
        with patch("llm_client.providers.async_providers.AsyncGroq", None):
            from llm_client.providers.async_providers import AsyncGroqProvider

            with pytest.raises(ProviderNotAvailableError):
                AsyncGroqProvider(llm="llama-3.3-70b-versatile", api_key="gsk-test")
