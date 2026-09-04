from unittest.mock import MagicMock, patch

import pytest
from groq import APIStatusError

from llm_client.providers.providers import GroqProvider


@pytest.fixture
def mock_groq_client():
    client = MagicMock()
    # Mocking chat.completions.create
    return client


def test_groq_fallback_sync(mock_groq_client):
    provider = GroqProvider(llm="qwen/qwen3-32b", api_key="fake-key")
    provider.client = mock_groq_client

    # Create the error
    error_response = MagicMock()
    error_response.status_code = 413
    error_message = "Rate limit exceeded on tokens per minute (TPM): Limit 1000, Requested 2000, please reduce your message size"

    # We need to make sure str(error) contains the message
    error = APIStatusError(
        message=error_message,
        response=error_response,
        body={"error": {"message": error_message, "type": "tokens", "code": "rate_limit_exceeded"}},
    )
    error.__str__ = lambda self: error_message

    # First call raises error, second call succeeds
    mock_response_success = MagicMock()
    mock_response_success.choices[0].message.content = "Success with fallback"

    mock_groq_client.chat.completions.create.side_effect = [error, mock_response_success]

    messages = [{"role": "user", "content": "Large request"}]
    with patch("llm_client.providers.providers.APIStatusError", APIStatusError):
        response = provider.chat_completion(messages)

    assert response == "Success with fallback"
    assert provider.llm == "meta-llama/llama-prompt-guard-2-22m"
    assert mock_groq_client.chat.completions.create.call_count == 2

    # Verify both calls had correct parameters
    first_call = mock_groq_client.chat.completions.create.call_args_list[0]
    assert first_call.kwargs["model"] == "qwen/qwen3-32b"

    second_call = mock_groq_client.chat.completions.create.call_args_list[1]
    assert second_call.kwargs["model"] == "meta-llama/llama-prompt-guard-2-22m"


@pytest.mark.asyncio
async def test_groq_fallback_async():
    from llm_client.providers.async_providers import AsyncGroqProvider

    provider = AsyncGroqProvider(llm="qwen/qwen3-32b", api_key="fake-key")
    mock_async_client = MagicMock()
    provider.client = mock_async_client

    error_response = MagicMock()
    error_response.status_code = 413
    error_message = "Rate limit exceeded on tokens per minute (TPM): Limit 1000, Requested 2000"

    error = APIStatusError(
        message=error_message,
        response=error_response,
        body={"error": {"message": error_message, "type": "tokens", "code": "rate_limit_exceeded"}},
    )
    error.__str__ = lambda self: error_message

    mock_response_success = MagicMock()
    mock_response_success.choices[0].message.content = "Async success with fallback"

    # Using AsyncMock for async methods
    from unittest.mock import AsyncMock

    mock_async_client.chat.completions.create = AsyncMock(
        side_effect=[error, mock_response_success]
    )

    messages = [{"role": "user", "content": "Large async request"}]
    from llm_client.providers import async_providers

    with patch.object(async_providers, "APIStatusError", APIStatusError):
        response = await provider.achat_completion(messages)

    assert response == "Async success with fallback"
    assert provider.llm == "meta-llama/llama-prompt-guard-2-22m"
    assert mock_async_client.chat.completions.create.call_count == 2
