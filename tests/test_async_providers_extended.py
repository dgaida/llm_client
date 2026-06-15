"""Erweiterte Tests für async providers zur Erhöhung der Code Coverage."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_client.exceptions import (
    ChatCompletionError,
)

pytestmark = pytest.mark.asyncio


class TestAsyncOpenAIProviderExtended:
    """Erweiterte Tests für AsyncOpenAIProvider."""

    async def test_achat_completion_with_files(self):
        """Test: Async chat completion with files for OpenAI."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Image analysis result"

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_openai.return_value = mock_client

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")
            messages = [{"role": "user", "content": "Analyze this"}]

            with patch(
                "llm_client.providers.async_providers.prepare_files_for_provider"
            ) as mock_prepare:
                mock_prepare.return_value = [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,fake_data"},
                    }
                ]

                response = await provider.achat_completion_with_files(messages, files=["test.jpg"])

                assert response == "Image analysis result"
                mock_prepare.assert_called_once_with(["test.jpg"], "openai")

    async def test_achat_completion_with_files_creates_new_message(self):
        """Test: File upload creates new message if last message is not user."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Analysis result"

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_openai.return_value = mock_client

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]

            with patch(
                "llm_client.providers.async_providers.prepare_files_for_provider"
            ) as mock_prepare:
                mock_prepare.return_value = [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,fake_data"},
                    }
                ]

                await provider.achat_completion_with_files(messages, files=["test.jpg"])

                # Verify create was called
                mock_client.chat.completions.create.assert_called_once()
                call_args = mock_client.chat.completions.create.call_args
                called_messages = call_args[1]["messages"]

                # Should have 3 messages now (original 2 + new file message)
                assert len(called_messages) == 3
                assert called_messages[-1]["role"] == "user"

    async def test_achat_completion_with_files_converts_string_content(self):
        """Test: Converts string content to list format when adding files."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Result"

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_openai.return_value = mock_client

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")
            messages = [{"role": "user", "content": "Analyze image"}]

            with patch(
                "llm_client.providers.async_providers.prepare_files_for_provider"
            ) as mock_prepare:
                mock_prepare.return_value = [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,fake_data"},
                    }
                ]

                await provider.achat_completion_with_files(messages, files=["test.jpg"])

                call_args = mock_client.chat.completions.create.call_args
                called_messages = call_args[1]["messages"]

                # Check that content was converted to list format
                last_message_content = called_messages[0]["content"]
                assert isinstance(last_message_content, list)
                assert len(last_message_content) == 2  # text + image
                assert last_message_content[0]["type"] == "text"
                assert last_message_content[0]["text"] == "Analyze image"

    async def test_achat_completion_with_files_client_not_initialized(self):
        """Test: Raises ChatCompletionError if client not initialized."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            mock_async_openai.return_value = MagicMock()

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")
            provider.client = None

            with pytest.raises(
                ChatCompletionError, match="RuntimeError: OpenAI client not initialized"
            ):
                await provider.achat_completion_with_files([], files=["test.jpg"])

    async def test_achat_completion_with_tools_without_tool_choice(self):
        """Test: Tool calling without explicit tool_choice parameter."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            mock_tool_call = MagicMock()
            mock_tool_call.id = "call_123"
            mock_tool_call.type = "function"
            mock_tool_call.function.name = "test_func"
            mock_tool_call.function.arguments = "{}"

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Result"
            mock_response.choices[0].message.tool_calls = [mock_tool_call]

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_openai.return_value = mock_client

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")
            tools = [{"type": "function", "function": {"name": "test_func"}}]

            await provider.achat_completion_with_tools([], tools, tool_choice=None)

            # Verify tool_choice was not passed if None
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert "tools" in call_kwargs
            # tool_choice should be included only if not None
            assert "tool_choice" not in call_kwargs


class TestAsyncGroqProviderExtended:
    """Erweiterte Tests für AsyncGroqProvider."""

    async def test_achat_completion_with_files(self):
        """Test: Async chat completion with files for Groq."""
        with patch("llm_client.providers.async_providers.AsyncGroq") as mock_async_groq:
            from llm_client.providers.async_providers import AsyncGroqProvider

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Groq image analysis"

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_groq.return_value = mock_client

            provider = AsyncGroqProvider(llm="llava-v1.5-7b-4096-preview", api_key="gsk-test")
            messages = [{"role": "user", "content": "Describe image"}]

            with patch("llm_client.providers.async_providers.detect_file_type") as mock_detect:
                mock_detect.return_value = "image"
                with patch(
                    "llm_client.providers.async_providers.prepare_files_for_provider"
                ) as mock_prepare:
                    mock_prepare.return_value = [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,fake_data"},
                        }
                    ]

                    response = await provider.achat_completion_with_files(
                        messages, files=["test.jpg"]
                    )

                    assert response == "Groq image analysis"
                    mock_detect.assert_called()

    async def test_achat_completion_with_files_non_image_raises_error(self):
        """Test: Groq raises ValueError for non-image files."""
        with patch("llm_client.providers.async_providers.AsyncGroq") as mock_async_groq:
            from llm_client.providers.async_providers import AsyncGroqProvider

            mock_async_groq.return_value = MagicMock()

            provider = AsyncGroqProvider(llm="llava-v1.5-7b-4096-preview", api_key="gsk-test")
            messages = [{"role": "user", "content": "Analyze"}]

            with patch("llm_client.providers.async_providers.detect_file_type") as mock_detect:
                mock_detect.return_value = "pdf"

                with pytest.raises(
                    ChatCompletionError, match="ValueError: Groq only supports image files"
                ):
                    await provider.achat_completion_with_files(messages, files=["test.pdf"])

    async def test_achat_completion_with_files_client_not_initialized(self):
        """Test: Groq raises ChatCompletionError if client not initialized."""
        with patch("llm_client.providers.async_providers.AsyncGroq") as mock_async_groq:
            from llm_client.providers.async_providers import AsyncGroqProvider

            mock_async_groq.return_value = MagicMock()

            provider = AsyncGroqProvider(llm="llava-v1.5-7b-4096-preview", api_key="gsk-test")
            provider.client = None

            with pytest.raises(
                ChatCompletionError, match="RuntimeError: Groq client not initialized"
            ):
                await provider.achat_completion_with_files([], files=["test.jpg"])

    async def test_achat_completion_client_not_initialized(self):
        """Test: achat_completion raises RuntimeError if client not initialized."""
        with patch("llm_client.providers.async_providers.AsyncGroq") as mock_async_groq:
            from llm_client.providers.async_providers import AsyncGroqProvider

            mock_async_groq.return_value = MagicMock()

            provider = AsyncGroqProvider(llm="llama-3.3-70b-versatile", api_key="gsk-test")
            provider.client = None

            with pytest.raises(ChatCompletionError, match="Groq client not initialized"):
                await provider.achat_completion([])

    async def test_achat_completion_stream_client_not_initialized(self):
        """Test: Streaming raises RuntimeError if client not initialized."""
        with patch("llm_client.providers.async_providers.AsyncGroq") as mock_async_groq:
            from llm_client.providers.async_providers import AsyncGroqProvider

            mock_async_groq.return_value = MagicMock()

            provider = AsyncGroqProvider(llm="llama-3.3-70b-versatile", api_key="gsk-test")
            provider.client = None

            with pytest.raises(ChatCompletionError, match="Groq client not initialized"):
                async for _ in provider.achat_completion_stream([]):
                    pass

    async def test_achat_completion_with_tools_client_not_initialized(self):
        """Test: Tool calling raises RuntimeError if client not initialized."""
        with patch("llm_client.providers.async_providers.AsyncGroq") as mock_async_groq:
            from llm_client.providers.async_providers import AsyncGroqProvider

            mock_async_groq.return_value = MagicMock()

            provider = AsyncGroqProvider(llm="llama-3.3-70b-versatile", api_key="gsk-test")
            provider.client = None

            with pytest.raises(ChatCompletionError, match="Groq client not initialized"):
                await provider.achat_completion_with_tools([], [])


class TestAsyncGeminiProviderExtended:
    """Erweiterte Tests für AsyncGeminiProvider."""

    async def test_achat_completion_with_files(self):
        """Test: Async chat completion with files for Gemini."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncGeminiProvider

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Gemini file analysis"

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_openai.return_value = mock_client

            provider = AsyncGeminiProvider(llm="gemini-3.1-flash-lite", api_key="AIzaSy-test")
            messages = [{"role": "user", "content": "Analyze"}]

            with patch(
                "llm_client.providers.async_providers.prepare_files_for_provider"
            ) as mock_prepare:
                mock_prepare.return_value = [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,fake_data"},
                    }
                ]

                response = await provider.achat_completion_with_files(messages, files=["test.jpg"])

                assert response == "Gemini file analysis"
                mock_prepare.assert_called_once_with(["test.jpg"], "gemini")

    async def test_achat_completion_with_files_client_not_initialized(self):
        """Test: Gemini raises ChatCompletionError if client not initialized."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncGeminiProvider

            mock_async_openai.return_value = MagicMock()

            provider = AsyncGeminiProvider(llm="gemini-3.1-flash-lite", api_key="AIzaSy-test")
            provider.client = None

            with pytest.raises(
                ChatCompletionError, match="RuntimeError: Gemini client not initialized"
            ):
                await provider.achat_completion_with_files([], files=["test.jpg"])

    async def test_achat_completion_client_not_initialized(self):
        """Test: achat_completion raises RuntimeError if client not initialized."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncGeminiProvider

            mock_async_openai.return_value = MagicMock()

            provider = AsyncGeminiProvider(llm="gemini-3.1-flash-lite", api_key="AIzaSy-test")
            provider.client = None

            with pytest.raises(ChatCompletionError, match="Gemini client not initialized"):
                await provider.achat_completion([])

    async def test_achat_completion_stream_client_not_initialized(self):
        """Test: Streaming raises RuntimeError if client not initialized."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncGeminiProvider

            mock_async_openai.return_value = MagicMock()

            provider = AsyncGeminiProvider(llm="gemini-3.1-flash-lite", api_key="AIzaSy-test")
            provider.client = None

            with pytest.raises(ChatCompletionError, match="Gemini client not initialized"):
                async for _ in provider.achat_completion_stream([]):
                    pass

    async def test_achat_completion_with_tools_client_not_initialized(self):
        """Test: Tool calling raises RuntimeError if client not initialized."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncGeminiProvider

            mock_async_openai.return_value = MagicMock()

            provider = AsyncGeminiProvider(llm="gemini-3.1-flash-lite", api_key="AIzaSy-test")
            provider.client = None

            with pytest.raises(ChatCompletionError, match="Gemini client not initialized"):
                await provider.achat_completion_with_tools([], [])


class TestAsyncProviderMixinExtended:
    """Erweiterte Tests für AsyncProviderMixin."""

    async def test_achat_completion_with_files_not_implemented(self):
        """Test: File upload raises NotImplementedError if not supported."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_async_openai.return_value = MagicMock()

            from llm_client.providers.async_providers import AsyncProviderMixin

            class TestProvider(AsyncProviderMixin):
                def __init__(self):
                    self.client = None

            provider = TestProvider()

            with pytest.raises(NotImplementedError, match="does not support file uploads"):
                await provider.achat_completion_with_files([], files=["test.jpg"])

    async def test_achat_completion_with_files_wraps_errors(self):
        """Test: File upload wraps errors in ChatCompletionError."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
            mock_async_openai.return_value = mock_client

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")

            with (
                patch("llm_client.providers.async_providers.prepare_files_for_provider"),
                pytest.raises(ChatCompletionError),
            ):
                await provider.achat_completion_with_files([], files=["test.jpg"])

    async def test_achat_completion_with_tools_wraps_errors(self):
        """Test: Tool calling wraps errors in ChatCompletionError."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
            mock_async_openai.return_value = mock_client

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")

            with pytest.raises(ChatCompletionError):
                await provider.achat_completion_with_tools([], [])

    async def test_streaming_not_implemented_error(self):
        """Test: Streaming raises correct error when not implemented."""
        from llm_client.exceptions import StreamingNotSupportedError
        from llm_client.providers.async_providers import AsyncProviderMixin

        class TestProvider(AsyncProviderMixin):
            def __init__(self):
                self.client = None

        provider = TestProvider()

        with pytest.raises(StreamingNotSupportedError):
            async for _ in provider.achat_completion_stream([]):
                pass


class TestAsyncOpenAIProviderRuntimeErrors:
    """Tests für RuntimeError-Fälle in AsyncOpenAIProvider."""

    async def test_achat_completion_stream_client_not_initialized(self):
        """Test: Streaming raises RuntimeError if client not initialized."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            mock_async_openai.return_value = MagicMock()

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")
            provider.client = None

            with pytest.raises(ChatCompletionError, match="OpenAI client not initialized"):
                async for _ in provider.achat_completion_stream([]):
                    pass

    async def test_achat_completion_client_not_initialized(self):
        """Test: achat_completion raises RuntimeError if client not initialized."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            mock_async_openai.return_value = MagicMock()

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")
            provider.client = None

            with pytest.raises(ChatCompletionError, match="OpenAI client not initialized"):
                await provider.achat_completion([])

    async def test_achat_completion_with_tools_client_not_initialized(self):
        """Test: Tool calling raises RuntimeError if client not initialized."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            mock_async_openai.return_value = MagicMock()

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")
            provider.client = None

            with pytest.raises(ChatCompletionError, match="OpenAI client not initialized"):
                await provider.achat_completion_with_tools([], [])


class TestToolCallsWithoutContent:
    """Tests für Tool Calls ohne Content."""

    async def test_openai_tool_calls_without_content(self):
        """Test: OpenAI tool calls when content is None."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncOpenAIProvider

            mock_tool_call = MagicMock()
            mock_tool_call.id = "call_abc"
            mock_tool_call.type = "function"
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = '{"arg": "value"}'

            mock_response = MagicMock()
            mock_response.choices[0].message.content = None
            mock_response.choices[0].message.tool_calls = [mock_tool_call]

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_openai.return_value = mock_client

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")
            tools = [{"type": "function", "function": {"name": "test_tool"}}]

            result = await provider.achat_completion_with_tools([], tools)

            assert result["content"] is None
            assert result["tool_calls"] is not None
            assert len(result["tool_calls"]) == 1

    async def test_groq_tool_calls_without_content(self):
        """Test: Groq tool calls when content is None."""
        with patch("llm_client.providers.async_providers.AsyncGroq") as mock_async_groq:
            from llm_client.providers.async_providers import AsyncGroqProvider

            mock_tool_call = MagicMock()
            mock_tool_call.id = "call_xyz"
            mock_tool_call.type = "function"
            mock_tool_call.function.name = "groq_tool"
            mock_tool_call.function.arguments = "{}"

            mock_response = MagicMock()
            mock_response.choices[0].message.content = None
            mock_response.choices[0].message.tool_calls = [mock_tool_call]

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_groq.return_value = mock_client

            provider = AsyncGroqProvider(llm="llama-3.3-70b-versatile", api_key="gsk-test")
            tools = [{"type": "function", "function": {"name": "groq_tool"}}]

            result = await provider.achat_completion_with_tools([], tools)

            assert result["content"] is None
            assert result["tool_calls"] is not None

    async def test_gemini_tool_calls_without_content(self):
        """Test: Gemini tool calls when content is None."""
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            from llm_client.providers.async_providers import AsyncGeminiProvider

            mock_tool_call = MagicMock()
            mock_tool_call.id = "call_123"
            mock_tool_call.type = "function"
            mock_tool_call.function.name = "gemini_tool"
            mock_tool_call.function.arguments = '{"data": "test"}'

            mock_response = MagicMock()
            mock_response.choices[0].message.content = None
            mock_response.choices[0].message.tool_calls = [mock_tool_call]

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_openai.return_value = mock_client

            provider = AsyncGeminiProvider(llm="gemini-3.1-flash-lite", api_key="AIzaSy-test")
            tools = [{"type": "function", "function": {"name": "gemini_tool"}}]

            result = await provider.achat_completion_with_tools([], tools)

            assert result["content"] is None
            assert result["tool_calls"] is not None
