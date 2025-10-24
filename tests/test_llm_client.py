import importlib
import sys

import pytest

from llm_client import LLMClient, llm_client


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Sorgt dafür, dass API Keys sauber getestet werden."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def test_auto_select_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = LLMClient()
    assert client.api_choice == "openai"
    assert client.llm.startswith("gpt")


def test_auto_select_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-test")
    client = LLMClient()
    assert client.api_choice == "groq"
    assert "moonshotai" in client.llm.lower()


# def test_auto_select_ollama(monkeypatch):
#     # Alle möglichen API Keys entfernen
#     monkeypatch.delenv("OPENAI_API_KEY", raising=False)
#     monkeypatch.delenv("GROQ_API_KEY", raising=False)
#
#     # dotenv.load_dotenv() so patchen, dass keine Datei geladen wird
#     monkeypatch.setattr(llm_client, "load_dotenv", lambda *args, **kwargs: None)
#
#     # Modul neu laden, um sicherzugehen, dass nichts gecached ist
#     importlib.reload(llm_client)
#
#     client = llm_client.LLMClient()
#
#     assert client.api_choice == "ollama"
#     assert "llama" in client.llm.lower()


def test_manual_override_to_ollama(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = LLMClient(api_choice="ollama")
    assert client.api_choice == "ollama"


def test_set_custom_model_and_temperature():
    client = LLMClient(llm="gpt-4o", temperature=0.5)
    assert client.llm == "gpt-4o"
    assert client.temperature == 0.5


def test_missing_ollama_package(monkeypatch):
    # Entferne ollama vollständig aus sys.modules
    sys.modules["ollama"] = None

    # Modul llm_client neu laden, damit der Import fehlschlägt
    importlib.reload(llm_client)

    # Jetzt Instanz mit ollama-API erstellen
    client = llm_client.LLMClient(api_choice="ollama")

    with pytest.raises(RuntimeError, match="Ollama Python package not available"):
        client.chat_completion([{"role": "user", "content": "test"}])

    # Cleanup: ollama wiederherstellen
    del sys.modules["ollama"]
    importlib.reload(llm_client)
