# Tests ausführen

Der LLM Client verwendet `pytest` für automatisierte Tests.

## Voraussetzungen

Stelle sicher, dass du die Entwicklungsabhängigkeiten installiert hast:

```bash
pip install -e ".[dev]"
```

## Tests ausführen

### Alle Tests

```bash
pytest
```

### Mit Coverage-Bericht

```bash
pytest --cov=llm_client --cov-report=term-missing
```

### Einzelne Testdatei

```bash
pytest tests/test_llm_client.py -v
```

## Test-Struktur

- `tests/test_llm_client.py`: Haupttests für den LLMClient
- `tests/test_providers.py`: Tests für individuelle Provider
- `tests/test_token_counter.py`: Tests für die Token-Zählung
- `tests/test_config.py`: Tests für das Laden von Konfigurationsdateien

## Mocking

Die Tests verwenden `unittest.mock`, um API-Aufrufe zu simulieren, sodass keine echten API-Keys benötigt werden und keine Kosten anfallen.
