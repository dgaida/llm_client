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

### 3. API Keys einrichten (optional, für Tests)

Erstellen Sie `secrets.env` im Projektverzeichnis:

```bash
# Optional: Für OpenAI-Tests
OPENAI_API_KEY=sk-xxxxxxxx

# Optional: Für Groq-Tests
GROQ_API_KEY=gsk-xxxxxxxx

# Optional: Für Gemini-Tests
GEMINI_API_KEY=AIzaSy-xxxxxxxx
```

**Hinweis**: Für die meisten Tests werden Mock-Objekte verwendet, echte API-Keys sind nicht erforderlich.

### 4. Pre-commit Hooks installieren (optional)
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

# Nur Gemini-Tests
pytest tests/test_llm_client.py -k "gemini" -v
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

**Beispiele**:  
- `feat(gemini): Add Google Gemini API support`  
- `fix(ollama): Handle connection timeout`  
- `docs(readme): Update API comparison table`  
- `test(gemini): Add unit tests for Gemini integration`  

## 🆕 Neue API hinzufügen

Wenn Sie eine neue LLM-API hinzufügen möchten:

1. **`llm_client.py` erweitern**:  
   - API-Key in `__init__` laden  
   - Default-Modell definieren  
   - Client-Initialisierung hinzufügen  
   - `chat_completion` Methode erweitern  

2. **Tests schreiben** (`tests/test_llm_client.py`):  
   - Initialisierungstests  
   - Chat-Completion Tests (gemockt)  
   - Error-Handling Tests  
   - Parameter-Validierung  

3. **Dokumentation aktualisieren**:  
   - `README.md`: API-Tabelle erweitern  
   - `README.md`: Beispiel hinzufügen  
   - `development/contributing.md`: Neue Abhängigkeiten dokumentieren  

4. **Beispiel**: Siehe Gemini-Integration als Referenz  

## 🐛 Bugs melden

Bitte nutzen Sie GitHub Issues und geben Sie an:  
- Python Version  
- Betriebssystem  
- Verwendete API (OpenAI, Groq, Gemini, Ollama)  
- Fehlerlog  
- Minimales Reproduktionsbeispiel  

## 💡 Feature Requests

Feature-Ideen sind willkommen! Bitte öffnen Sie ein Issue mit:  
- Beschreibung des Features  
- Use Case / Anwendungsfall  
- Mögliche Implementierung (optional)  
- Welche APIs betroffen sind  

## 🧩 Code-Struktur

### Hauptkomponenten

- **`llm_client/llm_client.py`**: Hauptklasse mit API-Logik  
- **`llm_client/adapter.py`**: llama-index Integration  
- **`tests/test_llm_client.py`**: Unit-Tests  
- **`tests/test_adapter.py`**: Adapter-Tests  

### Unterstützte APIs

| API | Client-Library | Base URL | Besonderheiten |
|-----|----------------|----------|----------------|
| OpenAI | `openai` | Standard | Native Integration |
| Groq | `groq` | Standard | Native Integration |
| Gemini | `openai` | `https://generativelanguage.googleapis.com/v1beta/openai/` | OpenAI-Kompatibilitätsmodus |
| Ollama | `ollama` | Lokal | Keine API-Keys nötig |

## 🔒 Sicherheit

- **Keine API-Keys im Code**: Nutzen Sie `secrets.env` oder Umgebungsvariablen  
- **Secrets nicht committen**: `.gitignore` prüfen  
- **Sensible Tests**: Mock-Objekte statt echte API-Calls  

## 📝 Dokumentations-Standards

- **Docstrings**: Google-Style für alle öffentlichen Methoden  
- **Type Hints**: Vollständige Typ-Annotationen  
- **Beispiele**: Code-Beispiele in Docstrings  
- **README**: Aktuell halten bei neuen Features  

## 🤝 Code Review

Pull Requests werden geprüft auf:  
- ✅ Tests laufen durch (pytest)  
- ✅ Code-Style korrekt (black, ruff)  
- ✅ Type-Hints vorhanden (mypy)  
- ✅ Dokumentation aktualisiert  
- ✅ Keine Secrets im Code  
- ✅ Sinnvolle Commit-Messages  

## 📄 Lizenz

Mit Ihrem Beitrag stimmen Sie zu, dass Ihre Änderungen unter der MIT-Lizenz veröffentlicht werden.

## 🙏 Danke!

Jeder Beitrag, ob groß oder klein, hilft das Projekt zu verbessern!
