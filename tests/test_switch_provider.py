"""Tests für die switch_provider Methode des LLMClient."""

from unittest.mock import MagicMock, patch

import pytest

from llm_client import LLMClient


class TestSwitchProvider:
    """Tests für die switch_provider Methode."""

    def test_switch_from_openai_to_gemini(self, monkeypatch):
        """Test: Wechsel von OpenAI zu Gemini."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")

        with patch("llm_client.llm_client.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Start mit OpenAI
            client = LLMClient(api_choice="openai")
            assert client.api_choice == "openai"
            assert "gpt" in client.llm

            # Wechsel zu Gemini
            client.switch_provider("gemini")
            assert client.api_choice == "gemini"
            assert "gemini" in client.llm.lower()

            # Prüfe dass OpenAI zweimal aufgerufen wurde (einmal für OpenAI, einmal für Gemini)
            assert mock_openai.call_count == 2

    def test_switch_from_groq_to_openai(self, monkeypatch):
        """Test: Wechsel von Groq zu OpenAI."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with (
            patch("llm_client.llm_client.Groq") as mock_groq,
            patch("llm_client.llm_client.OpenAI") as mock_openai,
        ):
            mock_groq_client = MagicMock()
            mock_openai_client = MagicMock()
            mock_groq.return_value = mock_groq_client
            mock_openai.return_value = mock_openai_client

            # Start mit Groq
            client = LLMClient(api_choice="groq")
            assert client.api_choice == "groq"

            # Wechsel zu OpenAI
            client.switch_provider("openai")
            assert client.api_choice == "openai"
            assert "gpt" in client.llm

    def test_switch_with_custom_model(self, monkeypatch):
        """Test: Provider-Wechsel mit benutzerdefiniertem Modell."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")

        with patch("llm_client.llm_client.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            client = LLMClient(api_choice="openai")

            # Wechsel zu Gemini mit spezifischem Modell
            client.switch_provider("gemini", llm="gemini-2.5-flash")
            assert client.api_choice == "gemini"
            assert client.llm == "gemini-2.5-flash"

    def test_switch_with_updated_temperature(self, monkeypatch):
        """Test: Provider-Wechsel mit neuer Temperatur."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        with (
            patch("llm_client.llm_client.OpenAI") as mock_openai,
            patch("llm_client.llm_client.Groq") as mock_groq,
        ):
            mock_openai.return_value = MagicMock()
            mock_groq.return_value = MagicMock()

            client = LLMClient(api_choice="openai", temperature=0.7)
            assert client.temperature == 0.7

            # Wechsel mit neuer Temperatur
            client.switch_provider("groq", temperature=0.3)
            assert client.api_choice == "groq"
            assert client.temperature == 0.3

    def test_switch_with_updated_max_tokens(self, monkeypatch):
        """Test: Provider-Wechsel mit neuen max_tokens."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.llm_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai", max_tokens=512)
            assert client.max_tokens == 512

            # Wechsel mit neuen max_tokens
            client.switch_provider("openai", max_tokens=2048)
            assert client.max_tokens == 2048

    def test_switch_with_all_parameters(self, monkeypatch):
        """Test: Provider-Wechsel mit allen optionalen Parametern."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")

        with patch("llm_client.llm_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(
                api_choice="openai", llm="gpt-3.5-turbo", temperature=0.7, max_tokens=512
            )

            # Wechsel mit allen Parametern
            client.switch_provider(
                "gemini", llm="gemini-2.5-flash", temperature=0.5, max_tokens=1024
            )

            assert client.api_choice == "gemini"
            assert client.llm == "gemini-2.5-flash"
            assert client.temperature == 0.5
            assert client.max_tokens == 1024

    def test_switch_to_ollama(self, monkeypatch):
        """Test: Wechsel zu Ollama (keine API-Keys nötig)."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.llm_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai")
            assert client.api_choice == "openai"

            # Wechsel zu Ollama
            client.switch_provider("ollama")
            assert client.api_choice == "ollama"
            assert client.client is None  # Ollama hat keinen Client

    def test_switch_invalid_provider_raises_error(self, monkeypatch):
        """Test: Ungültiger Provider wirft ValueError."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.llm_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai")

            with pytest.raises(ValueError, match="Invalid api_choice"):
                client.switch_provider("invalid_provider")

    def test_switch_without_api_key_raises_error(self, monkeypatch):
        """Test: Wechsel ohne API-Key wirft RuntimeError."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        with patch("llm_client.llm_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai")

            with pytest.raises(RuntimeError, match="GROQ_API_KEY not found"):
                client.switch_provider("groq")

    def test_switch_preserves_old_parameters(self, monkeypatch):
        """Test: Alte Parameter bleiben erhalten wenn nicht überschrieben."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

        with (
            patch("llm_client.llm_client.OpenAI") as mock_openai,
            patch("llm_client.llm_client.Groq") as mock_groq,
        ):
            mock_openai.return_value = MagicMock()
            mock_groq.return_value = MagicMock()

            client = LLMClient(api_choice="openai", temperature=0.8, max_tokens=1024)

            # Wechsel ohne Parameter-Änderung
            client.switch_provider("groq")

            # Alte Parameter sollten erhalten bleiben
            assert client.temperature == 0.8
            assert client.max_tokens == 1024

    def test_switch_case_insensitive(self, monkeypatch):
        """Test: Provider-Name ist case-insensitive."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")

        with patch("llm_client.llm_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            client = LLMClient(api_choice="openai")

            # Verschiedene Case-Variationen
            client.switch_provider("GEMINI")
            assert client.api_choice == "gemini"

            client.switch_provider("OpenAI")
            assert client.api_choice == "openai"

            client.switch_provider("OlLaMa")
            assert client.api_choice == "ollama"

    def test_switch_and_chat_completion(self, monkeypatch):
        """Test: Chat Completion funktioniert nach Provider-Wechsel."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")

        mock_openai_response = MagicMock()
        mock_openai_response.choices[0].message.content = "OpenAI response"

        mock_gemini_response = MagicMock()
        mock_gemini_response.choices[0].message.content = "Gemini response"

        with patch("llm_client.llm_client.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Start mit OpenAI
            client = LLMClient(api_choice="openai")
            mock_client.chat.completions.create.return_value = mock_openai_response

            messages = [{"role": "user", "content": "Hello"}]
            response1 = client.chat_completion(messages)
            assert response1 == "OpenAI response"

            # Wechsel zu Gemini
            client.switch_provider("gemini")
            mock_client.chat.completions.create.return_value = mock_gemini_response

            response2 = client.chat_completion(messages)
            assert response2 == "Gemini response"

    def test_switch_updates_default_model(self, monkeypatch):
        """Test: Default-Modell wird beim Provider-Wechsel aktualisiert."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")

        with (
            patch("llm_client.llm_client.OpenAI") as mock_openai,
            patch("llm_client.llm_client.Groq") as mock_groq,
        ):
            mock_openai.return_value = MagicMock()
            mock_groq.return_value = MagicMock()

            # Start mit OpenAI (Default-Modell)
            client = LLMClient(api_choice="openai")
            assert client.llm == "gpt-4o-mini"

            # Wechsel zu Groq ohne Modell-Angabe -> Default-Modell
            client.switch_provider("groq")
            assert "moonshotai" in client.llm.lower()

            # Wechsel zu Gemini ohne Modell-Angabe -> Default-Modell
            client.switch_provider("gemini")
            assert "gemini" in client.llm.lower()

            # Wechsel zu Ollama ohne Modell-Angabe -> Default-Modell
            client.switch_provider("ollama")
            assert "llama" in client.llm.lower()

    def test_switch_with_user_model_then_default(self, monkeypatch):
        """Test: Wechsel mit User-Modell, dann zurück zu Default."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("llm_client.llm_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()

            # Start mit benutzerdefiniertem Modell
            client = LLMClient(api_choice="openai", llm="gpt-4o")
            assert client.llm == "gpt-4o"

            # Wechsel mit neuem Modell
            client.switch_provider("openai", llm="gpt-3.5-turbo")
            assert client.llm == "gpt-3.5-turbo"

            # Wechsel ohne Modell -> Default
            client.switch_provider("openai")
            assert client.llm == "gpt-4o-mini"

    def test_multiple_switches(self, monkeypatch):
        """Test: Mehrere Provider-Wechsel hintereinander."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-test")

        with (
            patch("llm_client.llm_client.OpenAI") as mock_openai,
            patch("llm_client.llm_client.Groq") as mock_groq,
        ):
            mock_openai.return_value = MagicMock()
            mock_groq.return_value = MagicMock()

            client = LLMClient(api_choice="openai")

            # Mehrere Wechsel
            client.switch_provider("groq")
            assert client.api_choice == "groq"

            client.switch_provider("gemini")
            assert client.api_choice == "gemini"

            client.switch_provider("ollama")
            assert client.api_choice == "ollama"

            client.switch_provider("openai")
            assert client.api_choice == "openai"
