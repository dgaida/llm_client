"""Tests für den LLMClientAdapter."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from llm_client import LLMClient

# Mock llama_index falls nicht installiert
try:
    from llama_index.core.llms import ChatMessage, ChatResponse, LLMMetadata

    LLAMA_INDEX_INSTALLED = True
except ImportError:
    LLAMA_INDEX_INSTALLED = False
    # Mock-Klassen für Tests ohne llama_index
    ChatMessage = MagicMock
    ChatResponse = MagicMock
    LLMMetadata = MagicMock


@pytest.fixture
def mock_llm_client():
    """Erstellt einen Mock LLMClient für Tests."""
    client = MagicMock(spec=LLMClient)
    client.llm = "gpt-4o-mini"
    client.api_choice = "openai"
    client.temperature = 0.7
    client.chat_completion.return_value = "Test response from LLM"
    return client


@pytest.mark.skipif(not LLAMA_INDEX_INSTALLED, reason="llama-index-core not installed")
class TestLLMClientAdapterWithLlamaIndex:
    """Tests für LLMClientAdapter wenn llama-index installiert ist."""

    def test_adapter_initialization(self, mock_llm_client):
        """Test: Adapter kann mit Client initialisiert werden."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=mock_llm_client)
        assert adapter.client == mock_llm_client

    def test_adapter_without_client_raises_error(self):
        """Test: Fehler wenn kein Client übergeben wird."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter()
        with pytest.raises(ValueError, match="LLMClient instance must be provided"):
            _ = adapter.model

    def test_chat_converts_messages_correctly(self, mock_llm_client):
        """Test: Chat konvertiert llama_index Nachrichten korrekt."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=mock_llm_client)

        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi"),
        ]

        response = adapter.chat(messages)

        # Prüfe, dass chat_completion mit korrektem Format aufgerufen wurde
        mock_llm_client.chat_completion.assert_called_once_with(
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        )

        # Prüfe Response-Typ
        assert isinstance(response, ChatResponse)
        assert response.message.role == "assistant"
        assert response.message.content == "Test response from LLM"

    def test_model_property(self, mock_llm_client):
        """Test: model Property gibt korrekten Modellnamen zurück."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=mock_llm_client)
        assert adapter.model == "gpt-4o-mini"

    def test_metadata_property(self, mock_llm_client):
        """Test: metadata Property gibt LLMMetadata zurück."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=mock_llm_client)
        metadata = adapter.metadata

        assert isinstance(metadata, LLMMetadata)
        assert metadata.model_name == "gpt-4o-mini"
        assert metadata.is_chat_model is True
        assert metadata.context_window == 2048
        assert metadata.num_output == 512

    def test_complete_raises_not_implemented(self, mock_llm_client):
        """Test: complete() wirft NotImplementedError."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=mock_llm_client)
        with pytest.raises(NotImplementedError, match="complete not implemented"):
            adapter.complete("test prompt")

    def test_stream_chat_raises_not_implemented(self, mock_llm_client):
        """Test: stream_chat() wirft NotImplementedError."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=mock_llm_client)
        with pytest.raises(NotImplementedError, match="stream_chat not implemented"):
            adapter.stream_chat([])

    def test_stream_complete_raises_not_implemented(self, mock_llm_client):
        """Test: stream_complete() wirft NotImplementedError."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=mock_llm_client)
        with pytest.raises(NotImplementedError, match="stream_complete not implemented"):
            adapter.stream_complete("test")

    @pytest.mark.asyncio
    async def test_astream_chat_raises_not_implemented(self, mock_llm_client):
        """Test: astream_chat() wirft NotImplementedError."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=mock_llm_client)
        with pytest.raises(NotImplementedError, match="astream_chat not implemented"):
            await adapter.astream_chat([])

    @pytest.mark.asyncio
    async def test_astream_complete_raises_not_implemented(self, mock_llm_client):
        """Test: astream_complete() wirft NotImplementedError."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=mock_llm_client)
        with pytest.raises(NotImplementedError, match="astream_complete not implemented"):
            await adapter.astream_complete("test")

    @pytest.mark.asyncio
    async def test_achat_raises_not_implemented(self, mock_llm_client):
        """Test: achat() wirft NotImplementedError."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=mock_llm_client)
        with pytest.raises(NotImplementedError, match="achat not implemented"):
            await adapter.achat([])

    @pytest.mark.asyncio
    async def test_acomplete_raises_not_implemented(self, mock_llm_client):
        """Test: acomplete() wirft NotImplementedError."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=mock_llm_client)
        with pytest.raises(NotImplementedError, match="acomplete not implemented"):
            await adapter.acomplete("test")

    def test_repr(self, mock_llm_client):
        """Test: __repr__ gibt korrekte String-Repräsentation zurück."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=mock_llm_client)
        repr_str = repr(adapter)
        assert "LLMClientAdapter" in repr_str
        assert "client=" in repr_str


class TestLLMClientAdapterWithoutLlamaIndex:
    """Tests für LLMClientAdapter wenn llama-index NICHT installiert ist."""

    def test_import_error_without_llama_index(self, mock_llm_client):
        """Test: ImportError wenn llama-index nicht installiert ist."""
        # Mock die llama_index imports
        with patch.dict(sys.modules, {"llama_index.core.llms": None}):
            # Modul neu laden um Import-Fehler zu simulieren
            import importlib

            from llm_client import adapter

            importlib.reload(adapter)

            with pytest.raises(ImportError, match="llama-index-core is required"):
                adapter.LLMClientAdapter(client=mock_llm_client)

            # Modul wieder normal laden
            importlib.reload(adapter)


class TestLLMClientAdapterIntegration:
    """Integrationstests für den Adapter (falls llama-index verfügbar)."""

    @pytest.mark.skipif(not LLAMA_INDEX_INSTALLED, reason="llama-index-core not installed")
    def test_integration_with_real_client(self, monkeypatch):
        """Test: Integration mit echtem LLMClient (gemockt)."""
        from llm_client import LLMClient, LLMClientAdapter

        # Mock die API calls
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        client = LLMClient()
        adapter = LLMClientAdapter(client=client)

        # Mocke die chat_completion Methode
        with patch.object(client, "chat_completion", return_value="Mocked response"):
            messages = [ChatMessage(role="user", content="Test")]
            response = adapter.chat(messages)

            assert response.message.content == "Mocked response"
            assert response.message.role == "assistant"

    @pytest.mark.skipif(not LLAMA_INDEX_INSTALLED, reason="llama-index-core not installed")
    def test_empty_messages_handling(self, mock_llm_client):
        """Test: Handling von leeren Nachrichten."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=mock_llm_client)
        messages = []

        response = adapter.chat(messages)

        mock_llm_client.chat_completion.assert_called_once_with([])
        assert isinstance(response, ChatResponse)

    @pytest.mark.skipif(not LLAMA_INDEX_INSTALLED, reason="llama-index-core not installed")
    def test_multiple_message_types(self, mock_llm_client):
        """Test: Verschiedene Message-Typen werden korrekt konvertiert."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=mock_llm_client)

        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
            ChatMessage(role="user", content="How are you?"),
        ]

        adapter.chat(messages)

        expected_call = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

        mock_llm_client.chat_completion.assert_called_once_with(expected_call)
