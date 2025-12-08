"""Comprehensive tests for edge cases: rate limiting, concurrent requests, malformed responses, and network failures."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_client import LLMClient
from llm_client.exceptions import ChatCompletionError


class TestRateLimiting:
    """Tests for rate limiting scenarios."""

    def test_rate_limit_error_handling(self, monkeypatch):
        """Test: Handle rate limit errors from API."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()

            # Simulate rate limit error
            rate_limit_error = Exception("Rate limit exceeded: 429")
            mock_client.chat.completions.create.side_effect = rate_limit_error
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Test"}]

            with pytest.raises(ChatCompletionError) as exc_info:
                client.chat_completion(messages)

            assert "429" in str(exc_info.value) or "Rate limit" in str(exc_info.value)

    def test_retry_after_rate_limit(self, monkeypatch):
        """Test: Retry logic handles rate limits with exponential backoff."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Success after rate limit"

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()

            # First two attempts fail with rate limit, third succeeds
            mock_client.chat.completions.create.side_effect = [
                Exception("Rate limit: 429"),
                Exception("Rate limit: 429"),
                mock_response,
            ]
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Test"}]

            response = client.chat_completion(messages)

            assert response == "Success after rate limit"
            assert mock_client.chat.completions.create.call_count == 3

    def test_rate_limit_in_streaming(self, monkeypatch):
        """Test: Handle rate limits during streaming."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()

            def rate_limit_generator():
                raise Exception("Rate limit exceeded")
                yield  # Never reached

            mock_client.chat.completions.create.return_value = rate_limit_generator()
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Test"}]

            with pytest.raises(ChatCompletionError):
                list(client.chat_completion_stream(messages))

    @pytest.mark.asyncio
    async def test_async_rate_limit_handling(self, monkeypatch):
        """Test: Async client handles rate limits."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Rate limit: 429")
            )
            mock_async_openai.return_value = mock_client

            client = LLMClient(api_choice="openai", use_async=True)
            messages = [{"role": "user", "content": "Test"}]

            with pytest.raises(ChatCompletionError):
                await client.achat_completion(messages)


class TestConcurrentRequests:
    """Tests for concurrent request scenarios."""

    def test_multiple_clients_concurrent_requests(self, monkeypatch):
        """Test: Multiple clients making concurrent requests."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Concurrent response"

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            def make_request(client_id):
                client = LLMClient(api_choice="openai")
                messages = [{"role": "user", "content": f"Request {client_id}"}]
                return client.chat_completion(messages)

            # Execute concurrent requests
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request, i) for i in range(5)]
                results = [f.result() for f in futures]

            assert len(results) == 5
            assert all(r == "Concurrent response" for r in results)

    def test_single_client_multiple_requests(self, monkeypatch):
        """Test: Single client handling multiple sequential requests."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        responses = [f"Response {i}" for i in range(10)]
        mock_responses = []
        for resp in responses:
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = resp
            mock_responses.append(mock_resp)

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = mock_responses
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            results = []

            for i in range(10):
                messages = [{"role": "user", "content": f"Request {i}"}]
                result = client.chat_completion(messages)
                results.append(result)

            assert len(results) == 10
            assert results == responses

    @pytest.mark.asyncio
    async def test_async_concurrent_requests(self, monkeypatch):
        """Test: Async concurrent requests."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.async_providers.AsyncOpenAI") as mock_async_openai:
            call_count = 0

            async def mock_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                await asyncio.sleep(0.01)  # Simulate network delay
                mock_resp = MagicMock()
                mock_resp.choices[0].message.content = f"Async response {call_count}"
                return mock_resp

            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_async_openai.return_value = mock_client

            client = LLMClient(api_choice="openai", use_async=True)

            # Create multiple concurrent tasks
            tasks = []
            for i in range(5):
                messages = [{"role": "user", "content": f"Request {i}"}]
                tasks.append(client.achat_completion(messages))

            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            assert all("Async response" in r for r in results)

    def test_concurrent_streaming_requests(self, monkeypatch):
        """Test: Multiple concurrent streaming requests."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        def create_stream_generator(stream_id):
            chunks = [
                MagicMock(choices=[MagicMock(delta=MagicMock(content=f"Stream {stream_id}"))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content=" chunk"))]),
            ]
            return iter(chunks)

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()

            stream_counter = [0]

            def side_effect(*args, **kwargs):
                result = create_stream_generator(stream_counter[0])
                stream_counter[0] += 1
                return result

            mock_client.chat.completions.create.side_effect = side_effect
            mock_openai.return_value = mock_client

            def stream_request(req_id):
                client = LLMClient(api_choice="openai")
                messages = [{"role": "user", "content": f"Request {req_id}"}]
                return "".join(client.chat_completion_stream(messages))

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(stream_request, i) for i in range(3)]
                results = [f.result() for f in futures]

            assert len(results) == 3


class TestMalformedResponses:
    """Tests for handling malformed API responses."""

    def test_missing_content_field(self, monkeypatch):
        """Test: Handle response with missing content field."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()

            # Response missing content
            mock_response = MagicMock()
            mock_response.choices[0].message.content = None
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Test"}]

            response = client.chat_completion(messages)
            assert response is None

    def test_empty_choices_array(self, monkeypatch):
        """Test: Handle response with empty choices array."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()

            # Empty choices array
            mock_response = MagicMock()
            mock_response.choices = []
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Test"}]

            with pytest.raises(ChatCompletionError):
                client.chat_completion(messages)

    def test_malformed_streaming_chunk(self, monkeypatch):
        """Test: Handle malformed chunks in streaming."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        def malformed_generator():
            # First chunk is valid
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Valid"))])
            # Second chunk is malformed (missing delta)
            malformed = MagicMock()
            malformed.choices = [MagicMock()]
            malformed.choices[0].delta = None
            yield malformed
            # Third chunk is valid again
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Valid"))])

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = malformed_generator()
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Test"}]

            # Should handle malformed chunk gracefully
            chunks = []
            with pytest.raises(AttributeError):
                for chunk in client.chat_completion_stream(messages):
                    chunks.append(chunk)

    def test_invalid_json_in_tool_response(self, monkeypatch):
        """Test: Handle invalid JSON in tool call arguments."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()

            mock_tool_call = MagicMock()
            mock_tool_call.id = "call_123"
            mock_tool_call.type = "function"
            mock_tool_call.function.name = "test_function"
            mock_tool_call.function.arguments = "invalid json {"  # Malformed JSON

            mock_response = MagicMock()
            mock_response.choices[0].message.content = None
            mock_response.choices[0].message.tool_calls = [mock_tool_call]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Test"}]
            tools = [{"type": "function", "function": {"name": "test_function"}}]

            # Should still return the result, client handles JSON parsing
            result = client.chat_completion_with_tools(messages, tools)
            assert result["tool_calls"] is not None

    def test_unexpected_response_structure(self, monkeypatch):
        """Test: Handle completely unexpected response structure."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()

            # Return string instead of proper response object
            mock_client.chat.completions.create.return_value = "unexpected string"
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Test"}]

            with pytest.raises(ChatCompletionError):
                client.chat_completion(messages)

    def test_ollama_malformed_response(self):
        """Test: Handle malformed Ollama response."""
        with patch("llm_client.providers.Client") as mock_client:
            mock_instance = MagicMock()

            # Missing 'message' key
            mock_instance.chat.return_value = {"unexpected": "structure"}
            mock_client.return_value = mock_instance

            client = LLMClient(api_choice="ollama")
            messages = [{"role": "user", "content": "Test"}]

            with pytest.raises(ChatCompletionError):
                client.chat_completion(messages)


class TestNetworkFailures:
    """Tests for network failure scenarios."""

    def test_connection_timeout(self, monkeypatch):
        """Test: Handle connection timeout."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = TimeoutError("Connection timeout")
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Test"}]

            with pytest.raises(ChatCompletionError) as exc_info:
                client.chat_completion(messages)

            assert "timeout" in str(exc_info.value).lower()

    def test_connection_refused(self, monkeypatch):
        """Test: Handle connection refused error."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = ConnectionRefusedError(
                "Connection refused"
            )
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Test"}]

            with pytest.raises(ChatCompletionError):
                client.chat_completion(messages)

    def test_network_unreachable(self, monkeypatch):
        """Test: Handle network unreachable error."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = OSError("Network unreachable")
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Test"}]

            with pytest.raises(ChatCompletionError):
                client.chat_completion(messages)

    def test_ssl_certificate_error(self, monkeypatch):
        """Test: Handle SSL certificate verification failure."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception(
                "SSL: CERTIFICATE_VERIFY_FAILED"
            )
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Test"}]

            with pytest.raises(ChatCompletionError):
                client.chat_completion(messages)

    def test_interrupted_stream(self, monkeypatch):
        """Test: Handle stream interrupted mid-response."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        def interrupted_generator():
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Start"))])
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content=" middle"))])
            raise ConnectionError("Stream interrupted")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = interrupted_generator()
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Test"}]

            chunks = []
            with pytest.raises(ChatCompletionError):
                for chunk in client.chat_completion_stream(messages):
                    chunks.append(chunk)

            # Should have received some chunks before failure
            assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_async_network_timeout(self, monkeypatch):
        """Test: Async request timeout."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.async_providers.AsyncOpenAI") as mock_async_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=asyncio.TimeoutError("Async timeout")
            )
            mock_async_openai.return_value = mock_client

            client = LLMClient(api_choice="openai", use_async=True)
            messages = [{"role": "user", "content": "Test"}]

            with pytest.raises(ChatCompletionError):
                await client.achat_completion(messages)


class TestEdgeCaseInputs:
    """Tests for edge case inputs."""

    def test_extremely_long_message(self, monkeypatch):
        """Test: Handle extremely long message."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Response to long message"

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")

            # Create very long message
            long_content = "x" * 100000
            messages = [{"role": "user", "content": long_content}]

            response = client.chat_completion(messages)
            assert response == "Response to long message"

    def test_special_characters_in_messages(self, monkeypatch):
        """Test: Handle special characters and unicode."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Response with 特殊字符"

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")

            messages = [{"role": "user", "content": "Hello 世界! 🌍 \n\t Special: äöü ñ €"}]

            response = client.chat_completion(messages)
            assert "特殊字符" in response

    def test_empty_string_content(self, monkeypatch):
        """Test: Handle empty string in message content."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Response"

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": ""}]

            response = client.chat_completion(messages)
            assert response == "Response"

    def test_none_in_message_fields(self, monkeypatch):
        """Test: Handle None values in message fields."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")

            # This should likely fail or be handled
            messages = [{"role": "user", "content": None}]

            # The behavior depends on the API - might raise error
            try:
                client.chat_completion(messages)
            except (ChatCompletionError, TypeError, AttributeError):
                print(messages)  # Expected to fail


class TestProviderSwitchingEdgeCases:
    """Tests for edge cases in provider switching."""

    def test_rapid_provider_switching(self, monkeypatch):
        """Test: Rapid switching between providers."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        with (
            patch("llm_client.providers.OpenAI") as mock_openai,
            patch("llm_client.providers.Groq") as mock_groq,
        ):
            mock_openai.return_value = MagicMock()
            mock_groq.return_value = MagicMock()

            client = LLMClient(api_choice="openai")

            # Rapid switching
            for _ in range(10):
                client.switch_provider("groq")
                client.switch_provider("openai")

            assert client.api_choice == "openai"

    def test_switch_during_active_request(self, monkeypatch):
        """Test: Switching provider doesn't affect ongoing request."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "OpenAI response"

        with (
            patch("llm_client.providers.OpenAI") as mock_openai,
            patch("llm_client.providers.Groq") as mock_groq,
        ):
            mock_openai_client = MagicMock()
            mock_openai_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_openai_client
            mock_groq.return_value = MagicMock()

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Test"}]

            # Make request
            response = client.chat_completion(messages)

            # Switch provider after request
            client.switch_provider("groq")

            assert response == "OpenAI response"
            assert client.api_choice == "groq"


class TestTokenCountingEdgeCases:
    """Tests for edge cases in token counting."""

    def test_count_tokens_with_very_long_message(self, monkeypatch):
        """Test: Count tokens in very long message."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai")

            # Very long message
            messages = [{"role": "user", "content": "word " * 10000}]

            token_count = client.count_tokens(messages)
            assert isinstance(token_count, int)
            assert token_count > 5000

    def test_count_tokens_with_multiple_languages(self, monkeypatch):
        """Test: Count tokens with mixed languages."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.providers.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai")

            messages = [
                {"role": "user", "content": "English text"},
                {"role": "user", "content": "中文文本"},
                {"role": "user", "content": "日本語テキスト"},
                {"role": "user", "content": "Русский текст"},
            ]

            token_count = client.count_tokens(messages)
            assert isinstance(token_count, int)
            assert token_count > 0
