# 🧠 LLM Client

Ein universeller Python-Client zur Nutzung verschiedener Large Language Models (LLMs)
über **OpenAI**, [**Groq**](https://groq.com/), [**Google Gemini**](https://ai.google.dev/gemini-api) oder [**Ollama**](https://ollama.com/) – mit automatischer API-Erkennung.

---

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![codecov](https://codecov.io/gh/dgaida/llm_client/branch/master/graph/badge.svg)](https://codecov.io/gh/dgaida/llm_client)
[![Tests](https://github.com/dgaida/llm_client/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/llm_client/actions/workflows/tests.yml)
[![Code Quality](https://github.com/dgaida/llm_client/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/llm_client/actions/workflows/lint.yml)
[![CodeQL](https://github.com/dgaida/llm_client/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/llm_client/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## 📑 Inhaltsverzeichnis

- [Features](#-features)
- [Installation](#%EF%B8%8F-installation)
- [Schnellstart](#-schnellstart)
- [Verwendung](#-verwendung)
- [API Unterstützung](#-unterstützte-apis--default-modelle)
- [Tests](#-tests-ausführen)
- [Contributing](#-contributing)
- [Lizenz](#-lizenz)

## 🚀 Features

* 🔍 **Automatische API-Erkennung** - Nutzt verfügbare API-Keys oder fällt auf Ollama zurück
* ⚙️ **Einheitliches Interface** - Eine Methode für alle LLM-Backends
* 🔄 **Dynamischer Provider-Wechsel** - Wechsel zwischen APIs zur Laufzeit ohne neues Objekt
* 🧩 **Flexible Konfiguration** - Modell, Temperatur, Tokens frei wählbar
* 🧪 **Vollständige Tests** - Pytest-basiert mit hoher Code-Coverage
* 🔐 **Google Colab Support** - Automatisches Laden von Secrets aus userdata
* 🌟 **Google Gemini Support** - Nutzung via OpenAI-Kompatibilitätsmodus
* 📦 **Zero-Config** - Funktioniert out-of-the-box mit Ollama

---

## ⚙️ Installation

### Schnellinstallation

```bash
pip install git+https://github.com/dgaida/llm_client.git
```

### Entwicklungsinstallation

```bash
git clone https://github.com/dgaida/llm_client.git
cd llm_client
pip install -e ".[dev]"
```

### Mit llama-index Support

```bash
pip install -e ".[llama-index]"
```

---

## 🚦 Schnellstart

```python
from llm_client import LLMClient

# Automatische API-Erkennung
client = LLMClient()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Erkläre Machine Learning in einem Satz."}
]

response = client.chat_completion(messages)
print(response)
```

### Jupyter Notebook

Für einen Überblick über das Package teste das Jupyter Notebook [llm_client_example.ipynb](notebooks/llm_client_example.ipynb) auf Google Colab.

---

## 🔧 Konfiguration

### API-Keys einrichten

Erstellen Sie `secrets.env`:

```bash
# OpenAI
OPENAI_API_KEY=sk-xxxxxxxx

# Oder Groq
GROQ_API_KEY=gsk-xxxxxxxx

# Oder Google Gemini
GEMINI_API_KEY=AIzaSy-xxxxxxxx
```

**Ohne API-Keys**: Verwendet automatisch lokales [Ollama](https://ollama.com/) (Installation erforderlich).

### Google Colab

In Colab werden Keys automatisch aus `userdata` geladen:

```python
# Secrets → OPENAI_API_KEY, GROQ_API_KEY oder GEMINI_API_KEY hinzufügen
from llm_client import LLMClient
client = LLMClient()  # Lädt automatisch aus userdata
```

---

## 📚 Erweiterte Verwendung

### Spezifisches Modell wählen

```python
client = LLMClient(
    llm="gpt-4o",
    temperature=0.5,
    max_tokens=2048
)
```

### API manuell wählen

```python
# Gemini explizit wählen
client = LLMClient(api_choice="gemini", llm="gemini-2.5-flash")

# Ollama erzwingen (auch wenn API-Keys vorhanden)
client = LLMClient(api_choice="ollama")

# OpenAI explizit wählen
client = LLMClient(api_choice="openai", llm="gpt-4")
```

### Provider zur Laufzeit wechseln

**Neu in Version 0.2.0**: Sie können den LLM-Provider dynamisch wechseln, ohne ein neues `LLMClient`-Objekt erstellen zu müssen. Dies ist besonders nützlich für:

- Kostenoptimierung durch Wechsel zwischen verschiedenen APIs
- Fallback-Strategien bei API-Ausfällen
- A/B-Testing verschiedener Modelle
- Dynamische Provider-Auswahl basierend auf Anforderungen

```python
from llm_client import LLMClient

# Start mit OpenAI
client = LLMClient(api_choice="openai", llm="gpt-4o-mini")
response1 = client.chat_completion([{"role": "user", "content": "Hello"}])

# Wechsel zu Gemini
client.switch_provider("gemini", llm="gemini-2.5-flash")
response2 = client.chat_completion([{"role": "user", "content": "Hello"}])

# Wechsel zu Groq mit angepasster Temperatur
client.switch_provider("groq", temperature=0.3)
response3 = client.chat_completion([{"role": "user", "content": "Hello"}])

# Wechsel zu lokalem Ollama
client.switch_provider("ollama")
response4 = client.chat_completion([{"role": "user", "content": "Hello"}])
```

**Parameter von `switch_provider()`:**
- `api_choice` (erforderlich): Ziel-API ('openai', 'groq', 'gemini', 'ollama')
- `llm` (optional): Neues Modell. Wenn nicht angegeben, wird Default-Modell gewählt
- `temperature` (optional): Neue Temperatur. Wenn nicht angegeben, bleibt alte erhalten
- `max_tokens` (optional): Neue max_tokens. Wenn nicht angegeben, bleibt alte erhalten

**Beispiel: Fallback-Strategie**

```python
from llm_client import LLMClient

client = LLMClient(api_choice="openai")

try:
    response = client.chat_completion(messages)
except Exception as e:
    print(f"OpenAI failed: {e}")
    # Fallback zu Groq
    client.switch_provider("groq")
    response = client.chat_completion(messages)
```

**Beispiel: Kostenoptimierung**

```python
from llm_client import LLMClient

client = LLMClient()

# Günstigeres Modell für einfache Aufgaben
client.switch_provider("groq", llm="moonshotai/kimi-k2-instruct-0905")
simple_response = client.chat_completion(simple_messages)

# Leistungsstärkeres Modell für komplexe Aufgaben
client.switch_provider("openai", llm="gpt-4o")
complex_response = client.chat_completion(complex_messages)
```

### Gemini-Modelle nutzen

```python
from llm_client import LLMClient

# Automatisch, wenn GEMINI_API_KEY gesetzt ist
client = LLMClient()

# Oder explizit mit spezifischem Modell
client = LLMClient(
    api_choice="gemini",
    llm="gemini-2.5-flash",
    temperature=0.7
)

messages = [
    {"role": "user", "content": "Explain quantum computing"}
]
response = client.chat_completion(messages)
print(response)
```

### Mit llama-index Integration

```python
from llm_client import LLMClientAdapter, LLMClient

# Adapter erstellen (funktioniert auch mit Gemini)
llm_adapter = LLMClientAdapter(client=LLMClient(api_choice="gemini"))

# In llama-index verwenden
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, llm=llm_adapter)
```

---

## 🧩 Unterstützte APIs & Default-Modelle

| API    | Default-Modell                     | Bemerkung                           |
| ------ |------------------------------------|-------------------------------------|
| OpenAI | `gpt-4o-mini`                      | Schnell, zuverlässig                |
| Groq   | `moonshotai/kimi-k2-instruct-0905` | Sehr effizient auf GroqCloud        |
| Gemini | `gemini-2.0-flash-exp`             | Google's Modell via OpenAI-API      |
| Ollama | `llama3.2:1b`                      | Läuft lokal, kein API-Key nötig     |

### Ollama Installation

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download von https://ollama.ai/download

# Modell herunterladen
ollama pull llama3.2:1b
```

---

## 🧪 Tests ausführen

```bash
# Alle Tests
pytest

# Mit Coverage Report
pytest --cov=llm_client --cov-report=html

# Einzelne Tests
pytest tests/test_llm_client.py -v

# Tests für switch_provider Feature
pytest tests/test_switch_provider.py -v
```

### Code-Qualität prüfen

```bash
# Formatierung
black .

# Linting
ruff check .

# Auto-fix
ruff check --fix .
```

---

## 👥 Contributing

Beiträge sind willkommen! Siehe [CONTRIBUTING.md](CONTRIBUTING.md) für Details.

### Entwickler-Workflow

1. Fork & Clone
2. Feature-Branch erstellen: `git checkout -b feature/mein-feature`
3. Tests schreiben und ausführen
4. Code formatieren: `black . && ruff check --fix .`
5. Commit & Push
6. Pull Request öffnen

---

## 📊 Projektstruktur

```
llm_client/
├── .github/
│   └── workflows/         # CI/CD Pipelines
├── llm_client/
│   ├── __init__.py       # Package Exports
│   ├── llm_client.py     # Hauptklasse
│   └── adapter.py        # llama-index Integration
├── notebooks/
│   └── RAGChatbot_groq_API.ipynb
├── tests/
│   ├── test_llm_client.py
│   └── test_switch_provider.py
├── main.py               # Beispiel-Script
├── pyproject.toml        # Dependencies & Config
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## 📄 Lizenz

MIT License - siehe [LICENSE](LICENSE)

© 2025 Daniel Gaida, Technische Hochschule Köln

---

## 🔗 Weiterführende Links

* [Ollama Dokumentation](https://github.com/ollama/ollama)
* [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
* [Groq Cloud](https://groq.com/)
* [Google Gemini API](https://ai.google.dev/gemini-api/docs)
* [Gemini OpenAI Compatibility](https://ai.google.dev/gemini-api/docs/openai)
* [llama-index Docs](https://docs.llamaindex.ai/)
* [Andrew Ng's AISuite](https://github.com/andrewyng/aisuite)

---

## ⭐ Support

Wenn Ihnen dieses Projekt gefällt, geben Sie ihm einen Stern auf GitHub!

Fragen? Öffnen Sie ein [Issue](https://github.com/dgaida/llm_client/issues).
