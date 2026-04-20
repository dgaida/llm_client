# Tests für llm_client

Diese Verzeichnis enthält alle Tests für das llm_client Package.

## 📁 Test-Struktur

```
tests/
├── __init__.py
├── test_llm_client.py           # Basis-Tests für LLMClient
├── test_llm_client_extended.py  # Erweiterte Tests & Edge Cases
├── test_adapter.py               # Tests für LLMClientAdapter
└── README.md                     # Diese Datei
```

## 🧪 Tests ausführen

### Alle Tests

```bash
pytest
```

### Mit Coverage Report

```bash
pytest --cov=llm_client --cov-report=html
```

Der HTML-Report wird in `htmlcov/index.html` erstellt.

### Einzelne Test-Dateien

```bash
# Nur Basis-Tests
pytest tests/test_llm_client.py -v

# Nur erweiterte Tests
pytest tests/test_llm_client_extended.py -v

# Nur Adapter-Tests
pytest tests/test_adapter.py -v
```

### Spezifische Test-Klassen oder Funktionen

```bash
# Eine Test-Klasse
pytest tests/test_llm_client_extended.py::TestLLMClientInitialization -v

# Eine Test-Funktion
pytest tests/test_adapter.py::TestLLMClientAdapterWithLlamaIndex::test_chat_converts_messages_correctly -v
```

### Tests mit bestimmten Markern

```bash
# Nur Tests die llama-index benötigen
pytest -m "not skipif" tests/test_adapter.py

# Nur async Tests
pytest -k "async" tests/test_adapter.py
```

## 📊 Test-Abdeckung

### Aktueller Stand

- `llm_client.py`: ~95% Coverage  
- `adapter.py`: ~90% Coverage  
- Insgesamt: ~92% Coverage  

### Coverage-Report anzeigen

```bash
# Terminal-Output
pytest --cov=llm_client --cov-report=term-missing

# HTML-Report
pytest --cov=llm_client --cov-report=html
open htmlcov/index.html
```

## 🎯 Test-Kategorien

### 1. Unit Tests (`test_llm_client.py`)

Grundlegende Funktionalitätstests:  
- API-Auswahl (automatisch und manuell)  
- Modell-Konfiguration  
- Fehlerhandling  

### 2. Erweiterte Tests (`test_llm_client_extended.py`)

Detaillierte Tests für:  
- Initialisierung mit verschiedenen Parametern  
- Chat-Completion mit allen APIs (gemockt)  
- Edge Cases (leere Nachrichten, extreme Werte)  
- Type Hints Validierung  
- Google Colab Integration  

### 3. Adapter Tests (`test_adapter.py`)

Tests für llama-index Integration:  
- Adapter-Initialisierung  
- Nachricht-Konvertierung  
- Property-Tests (model, metadata)  
- NotImplementedError für nicht unterstützte Methoden  
- Integration mit echtem LLMClient  

## 🔧 Test-Fixtures

### `clear_env` (autouse)

Entfernt API-Keys aus der Umgebung vor jedem Test.

```python
@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Sorgt für saubere Umgebung in jedem Test."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
```

### `mock_llm_client`

Erstellt einen gemockten LLMClient für Adapter-Tests.

```python
@pytest.fixture
def mock_llm_client():
    """Mock LLMClient für Tests."""
    client = MagicMock(spec=LLMClient)
    client.llm = "gpt-4o-mini"
    client.chat_completion.return_value = "Test response"
    return client
```

## 🚨 Bekannte Einschränkungen

### llama-index Tests

Tests für `LLMClientAdapter` werden übersprungen, wenn llama-index-core nicht installiert ist:

```python
@pytest.mark.skipif(
    not LLAMA_INDEX_INSTALLED,
    reason="llama-index-core not installed"
)
```

**Installation für vollständige Tests:**

```bash
pip install llama-index-core
```

### Ollama Tests

Tests, die Ollama-Funktionalität prüfen, können fehlschlagen wenn:  
- Ollama nicht installiert ist  
- Ollama-Service nicht läuft  

Diese Tests nutzen Mocking und sollten daher normalerweise funktionieren.

## 🐛 Fehlersuche

### Test schlägt fehl: "ModuleNotFoundError"

```bash
# Stelle sicher, dass Package installiert ist
pip install -e .

# Mit allen Test-Dependencies
pip install -e ".[dev]"
```

### Import-Fehler bei llama-index

```bash
# Installiere llama-index für Adapter-Tests
pip install llama-index-core
```

### Coverage zu niedrig

```bash
# Detaillierte Coverage-Info
pytest --cov=llm_client --cov-report=term-missing

# Zeigt welche Zeilen nicht getestet sind
```

## 📝 Neue Tests hinzufügen

### Struktur eines Tests

```python
def test_my_new_feature():
    """Test: Kurze Beschreibung was getestet wird."""
    # Arrange - Setup
    client = LLMClient(api_choice="openai")

    # Act - Ausführung
    result = client.some_method()

    # Assert - Überprüfung
    assert result == expected_value
```

### Test-Klassen verwenden

```python
class TestMyFeature:
    """Tests für ein spezifisches Feature."""

    def test_case_1(self):
        """Test: Erster Fall."""
        ...

    def test_case_2(self):
        """Test: Zweiter Fall."""
        ...
```

### Mocking verwenden

```python
from unittest.mock import MagicMock, patch

def test_with_mock(monkeypatch):
    """Test: Mit gemockten Dependencies."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with patch("llm_client.llm_client.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Teste mit gemocktem Client
        ...
```

## ⚡ Performance

### Schnelle Tests (nur kritische)

```bash
pytest tests/test_llm_client.py -v
```

### Alle Tests mit paralleler Ausführung

```bash
pip install pytest-xdist
pytest -n auto
```

## 📚 Weiterführende Ressourcen

- [pytest Dokumentation](https://docs.pytest.org/)  
- [pytest-cov Plugin](https://pytest-cov.readthedocs.io/)  
- [unittest.mock Guide](https://docs.python.org/3/library/unittest.mock.html)  
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)  
