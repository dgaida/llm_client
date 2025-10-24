# Contributing to LLM Client

Vielen Dank für Ihr Interesse am LLM Client Projekt! 🎉

## 🚀 Entwicklungsumgebung einrichten

### 1. Repository klonen
```bash
git clone https://github.com/dgaida/llm_client.git
cd llm_client
```

### 2. Entwicklungsumgebung erstellen
```bash
# Mit Conda/Mamba
conda env create -f environment.yaml
conda activate llm-client-env

# Oder mit venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

pip install -e ".[dev]"
```

### 3. Pre-commit Hooks installieren (optional)
```bash
pip install pre-commit
pre-commit install
```

## 🧪 Tests ausführen

```bash
# Alle Tests
pytest

# Mit Coverage
pytest --cov=llm_client --cov-report=html

# Einzelne Test-Datei
pytest tests/test_llm_client.py -v
```

## 🎨 Code-Qualität

Vor dem Commit:

```bash
# Code formatieren
black .
ruff check --fix .

# Linting prüfen
ruff check .

# Type checking
mypy llm_client
```

## 📋 Pull Request Richtlinien

1. **Branch erstellen**: `git checkout -b feature/mein-feature`
2. **Tests schreiben**: Neue Features benötigen Tests
3. **Code formatieren**: Black & Ruff müssen durchlaufen
4. **Commit Messages**: Aussagekräftige Commit-Messages verwenden
5. **Pull Request öffnen**: Mit klarer Beschreibung der Änderungen

### Commit Message Format
```
type(scope): kurze Beschreibung

Längere Beschreibung bei Bedarf.

Fixes #123
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

## 🐛 Bugs melden

Bitte nutzen Sie GitHub Issues und geben Sie an:
- Python Version
- Betriebssystem
- Fehlerlog
- Minimales Reproduktionsbeispiel

## 💡 Feature Requests

Feature-Ideen sind willkommen! Bitte öffnen Sie ein Issue mit:
- Beschreibung des Features
- Use Case / Anwendungsfall
- Mögliche Implementierung (optional)

## 📄 Lizenz

Mit Ihrem Beitrag stimmen Sie zu, dass Ihre Änderungen unter der MIT-Lizenz veröffentlicht werden.
