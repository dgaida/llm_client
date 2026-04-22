"""Erweiterte Tests für LLMClient mit Gemini-Support."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from llm_client import LLMClient
from llm_client.exceptions import (
    InvalidProviderError,
    ProviderNotAvailableError,
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Sorgt dafür, dass API Keys sauber getestet werden."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


@pytest.fixture(autouse=True)
def set_dummy_openai_key(monkeypatch):
    """
    Setzt einen Dummy OPENAI_API_KEY für alle Tests automatisch,
    damit OpenAI-Client-Initialisierung nicht fehlschlägt.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-testdummy")


class TestLLMClientInitialization:
    """Tests für die Initialisierung des LLMClient."""

    def test_auto_select_openai(self, monkeypatch):
        """Test: OpenAI wird automatisch gewählt wenn API Key vorhanden."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        client = LLMClient()
        assert client.api_choice == "openai"
        assert client.llm.startswith("gpt")

    def test_auto_select_groq(self, monkeypatch):
        """Test: Groq wird automatisch gewählt wenn API Key vorhanden."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "groq-test")
        client = LLMClient()
        assert client.api_choice == "groq"
        assert "qwen" in client.llm.lower()

    def test_auto_select_gemini(self, monkeypatch):
        """Test: Gemini wird automatisch gewählt wenn API Key vorhanden."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")
        client = LLMClient()
        assert client.api_choice == "gemini"
        assert "gemini" in client.llm.lower()

    def test_manual_override_to_gemini(self, monkeypatch):
        """Test: API kann manuell auf Gemini gesetzt werden."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")
        client = LLMClient(api_choice="gemini")
        assert client.api_choice == "gemini"
        assert "gemini" in client.llm.lower()

    def test_manual_override_to_ollama(self, monkeypatch):
        """Test: API kann manuell auf Ollama gesetzt werden."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        client = LLMClient(api_choice="ollama")
        assert client.api_choice == "ollama"

    def test_manual_override_to_openai(self, monkeypatch):
        """Test: API kann manuell auf OpenAI gesetzt werden."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        client = LLMClient(api_choice="openai")
        assert client.api_choice == "openai"

    def test_invalid_api_choice_raises_error(self):
        """Test: Ungültige API-Wahl wirft ValueError."""
        with pytest.raises(
            InvalidProviderError,
            match="Invalid provider: invalid_api. Valid providers are: openai, groq, gemini, ollama",
        ):
            LLMClient(api_choice="invalid_api")

    def test_gemini_custom_model(self, monkeypatch):
        """Test: Benutzerdefiniertes Gemini-Modell."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")
        client = LLMClient(api_choice="gemini", llm="gemini-2.5-flash")
        assert client.llm == "gemini-2.5-flash"

    def test_set_custom_model_and_temperature(self):
        """Test: Benutzerdefiniertes Modell und Temperatur."""
        client = LLMClient(llm="gpt-4o", temperature=0.5)
        assert client.llm == "gpt-4o"
        assert client.temperature == 0.5

    def test_set_custom_max_tokens(self):
        """Test: Benutzerdefinierte max_tokens."""
        client = LLMClient(max_tokens=2048)
        assert client.max_tokens == 2048

    def test_set_custom_keep_alive(self):
        """Test: Benutzerdefinierter keep_alive Parameter."""
        client = LLMClient(keep_alive="10m")
        assert client.keep_alive == "10m"

    def test_gemini_client_initialization(self, monkeypatch):
        """Test: Gemini Client wird korrekt initialisiert."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test-key")

        # Patch at the correct location - in providers.py
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="gemini")

            assert client.gemini_api_key == "AIzaSy-test-key"
            assert client.api_choice == "gemini"

            # Prüfe dass OpenAI mit korrekter base_url initialisiert wurde
            mock_openai.assert_called_once_with(
                api_key="AIzaSy-test-key",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )

    def test_openai_client_initialization(self, monkeypatch):
        """Test: OpenAI Client wird korrekt initialisiert."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        client = LLMClient(api_choice="openai")
        assert client.openai_api_key == "sk-test-key"
        assert client.api_choice == "openai"

    def test_groq_client_initialization(self, monkeypatch):
        """Test: Groq Client wird korrekt initialisiert."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test-key")
        client = LLMClient(api_choice="groq")
        assert client.groq_api_key == "gsk-test-key"
        assert client.api_choice == "groq"


class TestLLMClientChatCompletion:
    """Tests für die chat_completion Methode."""

    def test_chat_completion_with_gemini(self, monkeypatch):
        """Test: chat_completion mit Gemini (gemockt)."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")

        # Mock Gemini response (via OpenAI compatibility)
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Gemini response"

        # Patch at the correct location - in providers.py
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="gemini")
            messages = [{"role": "user", "content": "Hello"}]
            response = client.chat_completion(messages)

            assert response == "Gemini response"
            mock_client.chat.completions.create.assert_called_once()

    def test_chat_completion_with_openai(self, monkeypatch):
        """Test: chat_completion mit OpenAI (gemockt)."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "OpenAI response"

        # Patch at the correct location - in providers.py
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [{"role": "user", "content": "Hello"}]
            response = client.chat_completion(messages)

            assert response == "OpenAI response"
            mock_client.chat.completions.create.assert_called_once()

    def test_chat_completion_with_groq(self, monkeypatch):
        """Test: chat_completion mit Groq (gemockt)."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        # Mock Groq response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Groq response"

        # Patch at the correct location - in providers.py
        with patch("llm_client.providers.providers.Groq") as mock_groq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_groq.return_value = mock_client

            client = LLMClient(api_choice="groq")
            messages = [{"role": "user", "content": "Hello"}]
            response = client.chat_completion(messages)

            assert response == "Groq response"
            mock_client.chat.completions.create.assert_called_once()

    def test_chat_completion_with_ollama(self):
        """Test: chat_completion mit Ollama (gemockt)."""
        mock_response = {"message": {"content": "Ollama response"}}

        # Patch at the correct location - in providers.py
        with patch("llm_client.providers.providers.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.chat.return_value = mock_response
            mock_client.return_value = mock_instance

            client = LLMClient(api_choice="ollama")
            messages = [{"role": "user", "content": "Hello"}]
            response = client.chat_completion(messages)

            assert response == "Ollama response"
            mock_instance.chat.assert_called_once()

    def test_chat_completion_without_gemini_client_raises_error(self, monkeypatch):
        """Test: RuntimeError wenn Gemini Client nicht verfügbar."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")

        # Patch at the correct location - in providers.py
        with (
            patch("llm_client.providers.providers.OpenAI", None),
            pytest.raises(
                ProviderNotAvailableError,
                match="gemini provider not available. Install with: pip install openai",
            ),
        ):
            LLMClient(api_choice="gemini")

    def test_chat_completion_without_openai_client_raises_error(self, monkeypatch):
        """Test: RuntimeError wenn OpenAI Client nicht verfügbar."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        # Patch at the correct location - in providers.py
        with (
            patch("llm_client.providers.providers.OpenAI", None),
            pytest.raises(
                ProviderNotAvailableError,
                match="openai provider not available. Install with: pip install openai",
            ),
        ):
            LLMClient(api_choice="openai")

    def test_chat_completion_without_groq_client_raises_error(self, monkeypatch):
        """Test: RuntimeError wenn Groq Client nicht verfügbar."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        # Patch at the correct location - in providers.py
        with (
            patch("llm_client.providers.providers.Groq", None),
            pytest.raises(
                ProviderNotAvailableError,
                match="groq provider not available. Install with: pip install groq",
            ),
        ):
            LLMClient(api_choice="groq")

    def test_missing_ollama_package(self):
        """Test: RuntimeError wenn Ollama Package nicht verfügbar."""
        # Patch at the correct location - in providers.py
        # Need to patch BOTH the import check AND prevent initialization
        with (
            patch("llm_client.providers.providers.Client", None),
            patch("llm_client.providers.providers.OLLAMA_AVAILABLE", False),
            pytest.raises(
                ProviderNotAvailableError,
                match="ollama provider not available",
            ),
        ):
            # Now the error should be raised during initialization
            LLMClient(api_choice="ollama")

    def test_gemini_parameters_passed_correctly(self, monkeypatch):
        """Test: Parameter werden korrekt an Gemini API übergeben."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Response"

        # Patch at the correct location - in providers.py
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(
                api_choice="gemini", llm="gemini-2.5-flash", temperature=0.8, max_tokens=1024
            )
            messages = [{"role": "user", "content": "Test"}]
            client.chat_completion(messages)

            # Prüfe ob Parameter korrekt übergeben wurden
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gemini-2.5-flash"
            assert call_args[1]["temperature"] == 0.8
            assert call_args[1]["max_tokens"] == 1024

    def test_chat_completion_parameters_passed_correctly(self, monkeypatch):
        """Test: Parameter werden korrekt an die API übergeben."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Response"

        # Patch at the correct location - in providers.py
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai", llm="gpt-4", temperature=0.8, max_tokens=1024)
            messages = [{"role": "user", "content": "Test"}]
            client.chat_completion(messages)

            # Prüfe ob Parameter korrekt übergeben wurden
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-4"
            assert call_args[1]["temperature"] == 0.8
            assert call_args[1]["max_tokens"] == 1024


class TestLLMClientEdgeCases:
    """Tests für Edge Cases und spezielle Szenarien."""

    def test_empty_messages_list(self, monkeypatch):
        """Test: Leere Nachrichten-Liste."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Response"

        # Patch at the correct location - in providers.py
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            response = client.chat_completion([])

            assert response == "Response"

    def test_multiple_messages(self, monkeypatch):
        """Test: Mehrere Nachrichten in der Konversation."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Final response"

        # Patch at the correct location - in providers.py
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")
            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
            ]
            response = client.chat_completion(messages)

            assert response == "Final response"
            # Prüfe dass alle Nachrichten übergeben wurden
            call_args = mock_client.chat.completions.create.call_args
            assert len(call_args[1]["messages"]) == 4

    def test_temperature_bounds(self):
        """Test: Verschiedene Temperatur-Werte."""
        # Sehr niedrige Temperatur
        client = LLMClient(temperature=0.0)
        assert client.temperature == 0.0

        # Hohe Temperatur
        client = LLMClient(temperature=2.0)
        assert client.temperature == 2.0

        # Standard-Temperatur
        client = LLMClient()
        assert client.temperature == 0.7

    def test_extreme_max_tokens(self):
        """Test: Extreme max_tokens Werte."""
        # Sehr kleine Werte
        client = LLMClient(max_tokens=1)
        assert client.max_tokens == 1

        # Sehr große Werte
        client = LLMClient(max_tokens=100000)
        assert client.max_tokens == 100000

    def test_repr_method(self, monkeypatch):
        """Test: __repr__ gibt korrekte String-Repräsentation zurück."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")
        client = LLMClient(api_choice="gemini", llm="gemini-2.5-flash", temperature=0.5)
        repr_str = repr(client)

        assert "LLMClient" in repr_str
        assert "gemini" in repr_str
        assert "gemini-2.5-flash" in repr_str
        assert "0.5" in repr_str

    def test_secrets_file_not_found(self, monkeypatch, tmp_path):
        """Test: Funktioniert wenn secrets.env nicht existiert."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        # Nutze einen nicht-existierenden Pfad
        fake_path = tmp_path / "nonexistent.env"

        client = LLMClient(secrets_path=str(fake_path))
        # Sollte auf Ollama zurückfallen
        assert client.api_choice == "ollama"

    def test_google_colab_integration_with_gemini(self, monkeypatch):
        """Test: Google Colab userdata Integration mit Gemini (gemockt)."""
        # Simuliere Colab-Umgebung
        monkeypatch.setitem(sys.modules, "google.colab", MagicMock())
        monkeypatch.setenv("COLAB_GPU", "1")

        # Mock userdata
        mock_userdata = MagicMock()
        mock_userdata.get.side_effect = lambda key: {
            "OPENAI_API_KEY": None,
            "GROQ_API_KEY": None,
            "GEMINI_API_KEY": "AIzaSy-colab-key",
        }.get(key)

        with patch.dict("sys.modules", {"google.colab": MagicMock(userdata=mock_userdata)}):
            client = LLMClient()
            print(client)

    def test_case_insensitive_api_choice(self, monkeypatch):
        """Test: api_choice ist case-insensitive."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        client1 = LLMClient(api_choice="OPENAI")
        assert client1.api_choice == "openai"

        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        client2 = LLMClient(api_choice="Groq")
        assert client2.api_choice == "groq"

        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")

        client3 = LLMClient(api_choice="GEMINI")
        assert client3.api_choice == "gemini"

        client4 = LLMClient(api_choice="OlLaMa")
        assert client4.api_choice == "ollama"


class TestLLMClientTypeHints:
    """Tests um sicherzustellen, dass Type Hints korrekt sind."""

    def test_messages_type_validation(self):
        """Test: Nachrichten haben korrektes Format."""
        # Korrektes Format
        valid_messages = [{"role": "user", "content": "Hello"}]

        # Diese sollten funktionieren
        assert isinstance(valid_messages, list)
        assert all(isinstance(m, dict) for m in valid_messages)
        assert all("role" in m and "content" in m for m in valid_messages)

    def test_return_type_is_string(self, monkeypatch):
        """Test: chat_completion gibt String zurück."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response"

        # Patch at the correct location - in providers.py
        with patch("llm_client.providers.providers.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="gemini")
            response = client.chat_completion([{"role": "user", "content": "Hi"}])

            assert isinstance(response, str)
