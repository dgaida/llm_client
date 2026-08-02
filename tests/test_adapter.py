"""Tests für den LLMClientAdapter."""

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


def is_llama_index_installed():
    """Check if llama-index is available."""
    try:
        import llama_index.core.llms  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not is_llama_index_installed(), reason="llama-index-core not installed")
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

        adapter_no_client = LLMClientAdapter(client=None)
        assert "client=None" in repr(adapter_no_client)

    def test_metadata_without_client_raises_error(self):
        """Test: metadata property raises error without client."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=None)
        with pytest.raises(ValueError, match="LLMClient instance must be provided"):
            _ = adapter.metadata

    def test_chat_without_client_raises_error(self):
        """Test: chat method raises error without client."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=None)
        with pytest.raises(ValueError, match="LLMClient instance must be provided"):
            adapter.chat([])


class TestLLMClientAdapterWithoutLlamaIndex:
    """Tests für LLMClientAdapter wenn llama-index NICHT installiert ist."""

    def test_import_error_without_llama_index(self, mock_llm_client):
        """Test: ImportError wenn llama-index nicht installiert ist."""
        from llm_client.providers import adapter

        with (
            patch("llm_client.providers.adapter.LLAMA_INDEX_AVAILABLE", False),
            pytest.raises(ImportError, match="llama-index-core is required"),
        ):
            adapter.LLMClientAdapter(client=mock_llm_client)


class TestLLMClientAdapterIntegration:
    """Integrationstests für den Adapter (falls llama-index verfügbar)."""

    @pytest.mark.skipif(not is_llama_index_installed(), reason="llama-index-core not installed")
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

    @pytest.mark.skipif(not is_llama_index_installed(), reason="llama-index-core not installed")
    def test_empty_messages_handling(self, mock_llm_client):
        """Test: Handling von leeren Nachrichten."""
        from llm_client import LLMClientAdapter

        adapter = LLMClientAdapter(client=mock_llm_client)
        messages = []

        response = adapter.chat(messages)

        mock_llm_client.chat_completion.assert_called_once_with([])
        assert isinstance(response, ChatResponse)

    @pytest.mark.skipif(not is_llama_index_installed(), reason="llama-index-core not installed")
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


class TestLLMClientAdapterMethodsMockedAllEnvironments:
    """Extra tests designed to run in all environments and cover all paths of LLMClientAdapter."""

    def test_all_adapter_methods_under_mock(self):
        """Test all methods of LLMClientAdapter with a mocked environment."""
        import importlib
        import sys
        import types

        from pydantic import BaseModel

        # Define mock classes that subclass BaseModel
        class DummyLLM(BaseModel):
            model_config = {"arbitrary_types_allowed": True}

        class MockChatMessage:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        class MockChatResponse:
            def __init__(self, message):
                self.message = message

        class MockLLMMetadata:
            def __init__(self, context_window, num_output, is_chat_model, model_name):
                self.context_window = context_window
                self.num_output = num_output
                self.is_chat_model = is_chat_model
                self.model_name = model_name

        dummy_llms = types.ModuleType("llama_index.core.llms")
        dummy_llms.LLM = DummyLLM
        dummy_llms.ChatMessage = MockChatMessage
        dummy_llms.ChatResponse = MockChatResponse
        dummy_llms.LLMMetadata = MockLLMMetadata
        dummy_llms.CompletionResponse = object

        # Force reload with our dummy llama_index modules
        try:
            with patch.dict(
                sys.modules,
                {
                    "llama_index": types.ModuleType("llama_index"),
                    "llama_index.core": types.ModuleType("llama_index.core"),
                    "llama_index.core.llms": dummy_llms,
                },
            ):
                from llm_client.providers import adapter

                importlib.reload(adapter)

                mock_client = MagicMock(spec=LLMClient)
                mock_client.llm = "gpt-4o-mini"
                mock_client.chat_completion.return_value = "Test response from LLM"

                # Test init
                adapter_instance = adapter.LLMClientAdapter(client=mock_client)
                assert adapter_instance.client is mock_client

                # Test chat
                messages = [MockChatMessage(role="user", content="Hello")]
                resp = adapter_instance.chat(messages)
                assert resp.message.content == "Test response from LLM"

                # Test model
                assert adapter_instance.model == "gpt-4o-mini"

                # Test metadata
                metadata = adapter_instance.metadata
                assert metadata.model_name == "gpt-4o-mini"

                # Test __repr__
                assert "LLMClientAdapter" in repr(adapter_instance)
                assert "client=" in repr(adapter_instance)

                # Test repr when client is None
                adapter_no_client = adapter.LLMClientAdapter(client=None)
                assert "client=None" in repr(adapter_no_client)

                # Test raised ValueErrors when client is None
                with pytest.raises(ValueError, match="LLMClient instance must be provided"):
                    _ = adapter_no_client.model

                with pytest.raises(ValueError, match="LLMClient instance must be provided"):
                    _ = adapter_no_client.metadata

                with pytest.raises(ValueError, match="LLMClient instance must be provided"):
                    adapter_no_client.chat([])

                # Test NotImplementedError raises
                with pytest.raises(NotImplementedError):
                    adapter_instance.complete("prompt")

                with pytest.raises(NotImplementedError):
                    adapter_instance.stream_chat([])

                with pytest.raises(NotImplementedError):
                    adapter_instance.stream_complete("prompt")

                # Test async NotImplementedError raises (running sync since they just return/raise)
                # But let's check with await as well since they are defined with async
                import asyncio

                with pytest.raises(NotImplementedError):
                    asyncio.run(adapter_instance.astream_chat([]))

                with pytest.raises(NotImplementedError):
                    asyncio.run(adapter_instance.astream_complete("prompt"))

                with pytest.raises(NotImplementedError):
                    asyncio.run(adapter_instance.achat([]))

                with pytest.raises(NotImplementedError):
                    asyncio.run(adapter_instance.acomplete("prompt"))

        finally:
            # Always restore back to original state outside of the mock context
            from llm_client.providers import adapter

            importlib.reload(adapter)
