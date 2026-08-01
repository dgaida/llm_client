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


class TestAsyncProvidersCoverageExpansion:
    """Extra tests specifically targeted to reach 100% test coverage on async_providers.py."""

    async def test_async_providers_import_errors(self):
        """Test import error fallback paths when packages are missing."""
        import sys
        import importlib
        from llm_client.providers import async_providers

        with patch.dict(sys.modules, {"openai": None, "groq": None}):
            importlib.reload(async_providers)
            assert async_providers.AsyncOpenAI is None
            assert async_providers.AsyncGroq is None

        # Restore original state
        importlib.reload(async_providers)

    async def test_async_provider_mixin_raises_not_implemented(self):
        """Test NotImplementedError in AsyncProviderMixin base class methods."""
        from llm_client.providers.async_providers import AsyncProviderMixin

        class DummyMixin(AsyncProviderMixin):
            pass

        mixin = DummyMixin()
        with pytest.raises(NotImplementedError):
            await mixin._achat_completion_impl([])

        with pytest.raises(NotImplementedError):
            await mixin._achat_completion_with_tools_impl([], [])

        with pytest.raises(NotImplementedError):
            await mixin._achat_completion_with_files_impl([])

    async def test_async_openai_extra_coverage(self):
        """Test remaining lines in AsyncOpenAIProvider."""
        from llm_client.providers.async_providers import AsyncOpenAIProvider

        # 1. tool_choice parameter branch (line 226)
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "response"
            mock_response.choices[0].message.tool_calls = None
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_openai.return_value = mock_client

            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")
            await provider.achat_completion_with_tools([], [], tool_choice="required")

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["tool_choice"] == "required"

        # 2. list_models uninitialized, success, and error paths (lines 322, 328)
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            provider = AsyncOpenAIProvider(llm="gpt-4o", api_key="sk-test")

            # Uninitialized client
            provider.client = None
            assert provider.list_models() == []

            # Success response
            provider.client = MagicMock()
            provider.client.api_key = "fake"

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [{"id": "model-1"}, {"id": "model-2"}]}

            with patch("requests.get", return_value=mock_response):
                assert provider.list_models() == ["model-1", "model-2"]

            # Failure response
            mock_response.status_code = 500
            with patch("requests.get", return_value=mock_response):
                assert provider.list_models() == []

    async def test_async_groq_extra_coverage(self):
        """Test remaining lines in AsyncGroqProvider."""
        from llm_client.providers.async_providers import AsyncGroqProvider
        from llm_client.exceptions import APIKeyNotFoundError

        # 1. APIKeyNotFoundError (line 342)
        with patch("llm_client.providers.async_providers.AsyncGroq", MagicMock()):
            with pytest.raises(APIKeyNotFoundError):
                AsyncGroqProvider(llm="groq-model", api_key=None)

        # 2. Sync RuntimeError (line 349)
        with patch("llm_client.providers.async_providers.AsyncGroq", MagicMock()):
            provider = AsyncGroqProvider(llm="groq-model", api_key="gsk-test")
            with pytest.raises(ChatCompletionError, match="AsyncGroqProvider only supports async"):
                provider.chat_completion([])

        # 3. Exception propagation in _achat_completion_impl (line 377)
        with patch("llm_client.providers.async_providers.AsyncGroq") as mock_async_groq:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=ValueError("Test Exception"))
            mock_async_groq.return_value = mock_client
            provider = AsyncGroqProvider(llm="groq-model", api_key="gsk-test")
            with pytest.raises(ChatCompletionError, match="Test Exception"):
                await provider.achat_completion([])

        # 4. Tool choice branch (line 409)
        with patch("llm_client.providers.async_providers.AsyncGroq") as mock_async_groq:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "resp"
            mock_response.choices[0].message.tool_calls = None
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_groq.return_value = mock_client
            provider = AsyncGroqProvider(llm="groq-model", api_key="gsk-test")
            await provider.achat_completion_with_tools([], [], tool_choice="auto")
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["tool_choice"] == "auto"

        # 5. File messages user-appending when messages are empty (line 458)
        with patch("llm_client.providers.async_providers.AsyncGroq") as mock_async_groq:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "resp"
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_groq.return_value = mock_client
            provider = AsyncGroqProvider(llm="groq-model", api_key="gsk-test")

            with (
                patch("llm_client.providers.async_providers.detect_file_type", return_value="image"),
                patch("llm_client.providers.async_providers.prepare_files_for_provider", return_value=[{"file": "data"}])
            ):
                # Empty messages list
                await provider.achat_completion_with_files([], files=["img.jpg"])
                call_args = mock_client.chat.completions.create.call_args[1]["messages"]
                assert len(call_args) == 1
                assert call_args[0]["role"] == "user"

        # 6. list_models uninitialized, success, failure (lines 510, 516)
        with patch("llm_client.providers.async_providers.AsyncGroq", MagicMock()):
            provider = AsyncGroqProvider(llm="groq-model", api_key="gsk-test")
            provider.client = None
            assert provider.list_models() == []

            provider.client = MagicMock()
            provider.client.api_key = "gsk-test"

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [{"id": "groq-1"}]}
            with patch("requests.get", return_value=mock_response):
                assert provider.list_models() == ["groq-1"]

            mock_response.status_code = 500
            with patch("requests.get", return_value=mock_response):
                assert provider.list_models() == []

    async def test_async_gemini_extra_coverage(self):
        """Test remaining lines in AsyncGeminiProvider."""
        from llm_client.providers.async_providers import AsyncGeminiProvider
        from llm_client.exceptions import ProviderNotAvailableError, APIKeyNotFoundError

        # 1. ProviderNotAvailableError (line 526)
        with patch("llm_client.providers.async_providers.AsyncOpenAI", None):
            with pytest.raises(ProviderNotAvailableError):
                AsyncGeminiProvider(llm="gemini", api_key="AIzaSy")

        # 2. APIKeyNotFoundError (line 530)
        with patch("llm_client.providers.async_providers.AsyncOpenAI", MagicMock()):
            with pytest.raises(APIKeyNotFoundError):
                AsyncGeminiProvider(llm="gemini", api_key=None)

        # 3. Sync RuntimeError (line 540)
        with patch("llm_client.providers.async_providers.AsyncOpenAI", MagicMock()):
            provider = AsyncGeminiProvider(llm="gemini", api_key="AIzaSy")
            with pytest.raises(ChatCompletionError, match="AsyncGeminiProvider only supports async"):
                provider.chat_completion([])

        # 4. Tool choice branch (line 585)
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "resp"
            mock_response.choices[0].message.tool_calls = None
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_openai.return_value = mock_client
            provider = AsyncGeminiProvider(llm="gemini", api_key="AIzaSy")
            await provider.achat_completion_with_tools([], [], tool_choice="required")
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["tool_choice"] == "required"

        # 5. File messages user-appending when messages are empty (line 629)
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "resp"
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_openai.return_value = mock_client
            provider = AsyncGeminiProvider(llm="gemini", api_key="AIzaSy")

            with patch("llm_client.providers.async_providers.prepare_files_for_provider", return_value=[{"file": "data"}]):
                await provider.achat_completion_with_files([], files=["img.jpg"])
                call_args = mock_client.chat.completions.create.call_args[1]["messages"]
                assert len(call_args) == 1
                assert call_args[0]["role"] == "user"

        # 6. list_models uninitialized, success, failure (lines 681, 687)
        with patch("llm_client.providers.async_providers.AsyncOpenAI", MagicMock()):
            provider = AsyncGeminiProvider(llm="gemini", api_key="AIzaSy")
            provider.client = None
            assert provider.list_models() == []

            provider.client = MagicMock()
            provider.client.api_key = "AIzaSy"

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [{"id": "gemini-1"}]}
            with patch("requests.get", return_value=mock_response):
                assert provider.list_models() == ["gemini-1"]

            mock_response.status_code = 500
            with patch("requests.get", return_value=mock_response):
                assert provider.list_models() == []

    async def test_async_kiconnect_extra_coverage(self):
        """Test remaining lines in AsyncKIConnectProvider."""
        from llm_client.providers.async_providers import AsyncKIConnectProvider
        from llm_client.exceptions import ProviderNotAvailableError, APIKeyNotFoundError

        # 1. ProviderNotAvailableError (line 697)
        with patch("llm_client.providers.async_providers.AsyncOpenAI", None):
            with pytest.raises(ProviderNotAvailableError):
                AsyncKIConnectProvider(llm="kiconnect", api_key="key")

        # 2. APIKeyNotFoundError (line 701)
        with patch("llm_client.providers.async_providers.AsyncOpenAI", MagicMock()):
            with pytest.raises(APIKeyNotFoundError):
                AsyncKIConnectProvider(llm="kiconnect", api_key=None)

        # 3. Sync RuntimeError (line 711)
        with patch("llm_client.providers.async_providers.AsyncOpenAI", MagicMock()):
            provider = AsyncKIConnectProvider(llm="kiconnect", api_key="key")
            with pytest.raises(ChatCompletionError, match="AsyncKIConnectProvider only supports async"):
                provider.chat_completion([])

        # 4. Client not initialized RuntimeError in _achat_completion_impl (line 718)
        with patch("llm_client.providers.async_providers.AsyncOpenAI", MagicMock()):
            provider = AsyncKIConnectProvider(llm="kiconnect", api_key="key")
            provider.client = None
            with pytest.raises(ChatCompletionError, match="KI Connect client not initialized"):
                await provider.achat_completion([])

        # 5. extra_content branch (line 733)
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "resp"
            mock_response.choices[0].message.extra_content = "some extra thought signature"
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_async_openai.return_value = mock_client
            provider = AsyncKIConnectProvider(llm="kiconnect", api_key="key")
            res = await provider.achat_completion([])
            assert res == "resp"

        # 6. Stream client uninitialized RuntimeError (line 741)
        with patch("llm_client.providers.async_providers.AsyncOpenAI", MagicMock()):
            provider = AsyncKIConnectProvider(llm="kiconnect", api_key="key")
            provider.client = None
            with pytest.raises(ChatCompletionError, match="KI Connect client not initialized"):
                async for _ in provider.achat_completion_stream([]):
                    pass

        # 7. Stream content yielding chunk delta content (line 754)
        with patch("llm_client.providers.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_client = MagicMock()

            async def dummy_generator():
                yield MagicMock(choices=[MagicMock(delta=MagicMock(content="token1"))])
                yield MagicMock(choices=[MagicMock(delta=MagicMock(content="token2"))])

            mock_client.chat.completions.create = AsyncMock(return_value=dummy_generator())
            mock_async_openai.return_value = mock_client
            provider = AsyncKIConnectProvider(llm="kiconnect", api_key="key")

            res = []
            async for chunk in provider.achat_completion_stream([]):
                res.append(chunk)
            assert res == ["token1", "token2"]

        # 8. list_models uninitialized, success, failure, raising error (lines 769, 776, 780)
        with patch("llm_client.providers.async_providers.AsyncOpenAI", MagicMock()):
            provider = AsyncKIConnectProvider(llm="kiconnect", api_key="key")

            # uninitialized
            provider.client = None
            assert provider.list_models() == []

            # success
            provider.client = MagicMock()
            provider.client.api_key = "key"
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [{"id": "kic-1"}]}
            with patch("requests.get", return_value=mock_response):
                assert provider.list_models() == ["kic-1"]

            # failure response
            mock_response.status_code = 500
            with patch("requests.get", return_value=mock_response):
                assert provider.list_models() == []

            # raising Exception
            with patch("requests.get", side_effect=Exception("network issue")):
                assert provider.list_models() == []

    async def test_async_kiconnect_get_default_model(self):
        """Test: AsyncKIConnectProvider.get_default_model() returns correct default model."""
        from llm_client.providers.async_providers import AsyncKIConnectProvider
        assert AsyncKIConnectProvider.get_default_model() == "openai-gpt5.5"

    async def test_async_groq_fallback_retry(self):
        """Test: AsyncGroqProvider fallback retry logic when rate limit is exceeded."""
        from llm_client.providers.async_providers import AsyncGroqProvider
        from groq import APIStatusError

        provider = AsyncGroqProvider(llm="qwen/qwen3-32b", api_key="gsk-test")
        mock_client = MagicMock()
        provider.client = mock_client

        error_response = MagicMock()
        error_response.status_code = 413
        error_message = "Rate limit exceeded on tokens per minute (TPM): Limit 10000, Requested 21142"

        error = APIStatusError(
            message=error_message,
            response=error_response,
            body={"error": {"message": error_message, "type": "tokens", "code": "rate_limit_exceeded"}},
        )
        error.__str__ = lambda self: error_message

        mock_response_success = MagicMock()
        mock_response_success.choices[0].message.content = "Async success with fallback"

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[error, mock_response_success]
        )

        messages = [{"role": "user", "content": "Large request"}]

        # We need to mock GroqProvider._find_fallback_model to return a model
        from llm_client.providers.providers import GroqProvider
        with patch.object(GroqProvider, "_find_fallback_model", return_value="meta-llama/llama-4-scout-17b-16e-instruct"):
            res = await provider.achat_completion(messages)
            assert res == "Async success with fallback"
            assert provider.llm == "meta-llama/llama-4-scout-17b-16e-instruct"

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
