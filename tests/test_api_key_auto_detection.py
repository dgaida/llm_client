import os
import sys
from unittest.mock import patch, MagicMock
import pytest
from llm_client import LLMClient
from llm_client.providers.provider_factory import ProviderFactory
from llm_client.providers.providers import OpenAIProvider, GroqProvider, GeminiProvider

def test_groq_default_model_updated():
    """Test that Groq default model is now qwen/qwen3-32b."""
    assert GroqProvider.get_default_model() == "qwen/qwen3-32b"

def test_detect_provider_from_key():
    """Test ProviderFactory.detect_provider_from_key logic."""
    assert ProviderFactory.detect_provider_from_key("sk-12345") == "openai"
    assert ProviderFactory.detect_provider_from_key("gsk-12345") == "groq"
    assert ProviderFactory.detect_provider_from_key("gsk_12345") == "groq"
    assert ProviderFactory.detect_provider_from_key("AIza12345") == "gemini"
    assert ProviderFactory.detect_provider_from_key("unknown-key") is None
    assert ProviderFactory.detect_provider_from_key("") is None

@patch.dict(os.environ, {"API_KEY": "gsk-test-key"})
@patch("llm_client.providers.providers.Groq")
def test_llm_client_auto_detects_groq_from_api_key(mock_groq):
    """Test that LLMClient auto-detects Groq when only API_KEY is provided."""
    # Ensure other keys are NOT in environment for this test
    with patch.dict(os.environ, {}, clear=True):
        with patch.dict(os.environ, {"API_KEY": "gsk-test-key"}):
            client = LLMClient()
            assert client.api_choice == "groq"
            assert client.provider.llm == "qwen/qwen3-32b"

@patch.dict(os.environ, {"API_KEY": "sk-test-key"})
@patch("llm_client.providers.providers.OpenAI")
def test_llm_client_auto_detects_openai_from_api_key(mock_openai):
    """Test that LLMClient auto-detects OpenAI when only API_KEY is provided."""
    with patch.dict(os.environ, {}, clear=True):
        with patch.dict(os.environ, {"API_KEY": "sk-test-key"}):
            client = LLMClient()
            assert client.api_choice == "openai"
            assert client.provider.llm == "gpt-4o-mini"

@patch.dict(os.environ, {"API_KEY": "AIza-test-key"})
@patch("llm_client.providers.providers.OpenAI") # Gemini uses OpenAI client
def test_llm_client_auto_detects_gemini_from_api_key(mock_openai):
    """Test that LLMClient auto-detects Gemini when only API_KEY is provided."""
    with patch.dict(os.environ, {}, clear=True):
        with patch.dict(os.environ, {"API_KEY": "AIza-test-key"}):
            client = LLMClient()
            assert client.api_choice == "gemini"
            assert client.provider.llm == "gemini-2.0-flash-exp"

def test_llm_client_colab_api_key_detection():
    """Test LLMClient detects API_KEY from Colab userdata."""
    mock_userdata = MagicMock()
    mock_userdata.get.side_effect = lambda key: "sk-colab" if key == "API_KEY" else Exception("Not found")

    mock_colab = MagicMock()
    mock_colab.userdata = mock_userdata

    with patch.dict("sys.modules", {"google.colab": mock_colab}):
        with patch.dict(os.environ, {"COLAB_GPU": "1"}):
            with patch.dict(os.environ, {}, clear=True):
                # We need to patch OpenAI to avoid actual API call during init if it tries to check availability
                with patch("llm_client.providers.providers.OpenAI"):
                    client = LLMClient()
                    assert client.api_key == "sk-colab"
                    assert client.api_choice == "openai"
