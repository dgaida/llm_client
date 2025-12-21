# 🧠 LLM Client

Ein universeller Python-Client zur Nutzung verschiedener Large Language Models (LLMs) über **OpenAI**, [**Groq**](https://groq.com/), [**Google Gemini**](https://ai.google.dev/gemini-api) oder [**Ollama**](https://ollama.com/) – mit automatischer API-Erkennung, dynamischem Provider-Wechsel, Token-Zählung, Async-Unterstützung und Konfigurationsdatei-Verwaltung.

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
- [Neu in v0.3.0](#-neu-in-v030)
- [Installation](#%EF%B8%8F-installation)
- [Schnellstart](#-schnellstart)
- [Verwendung](#-verwendung)
  - [Token-Zählung](#-token-zählung)
  - [Async-Unterstützung](#-async-unterstützung)
  - [Konfigurationsdateien](#-konfigurationsdateien)
  - [Response-Streaming](#-response-streaming)
  - [Dynamischer Provider-Wechsel](#-dynamischer-provider-wechsel)
- [Unterstützte APIs](#-unterstützte-apis--default-modelle)
- [Tests](#-tests-ausführen)
- [Architektur](#-projekt-architektur)
- [Contributing](#-contributing)
- [Lizenz](#-lizenz)

## 🚀 Features

### Kern-Features
* 🔍 **Automatische API-Erkennung** - Nutzt verfügbare API-Keys oder fällt auf Ollama zurück
* ⚙️ **Einheitliches Interface** - Eine Methode für alle LLM-Backends
* 🔄 **Dynamischer Provider-Wechsel** - Wechsel zwischen APIs zur Laufzeit ohne neues Objekt
* 🧩 **Flexible Konfiguration** - Modell, Temperatur, Tokens frei wählbar
* 🔐 **Google Colab Support** - Automatisches Laden von Secrets aus userdata
* 📦 **Zero-Config** - Funktioniert out-of-the-box mit Ollama

### Architektur
* 🏗️ **Strategy Pattern** - Saubere Architektur mit Provider-Klassen
* 🏭 **Factory Pattern** - Zentrale Provider-Erstellung und -Verwaltung
* 🧪 **Vollständige Tests** - Pytest-basiert mit >92% Code-Coverage
* 🌟 **Google Gemini Support** - Nutzung via OpenAI-Kompatibilitätsmodus

---

## ✨ Neu in v0.3.0

Version 0.3.0 führt vier große Features ein:

### 1. 📊 Token-Zählung mit tiktoken

```python
from llm_client import LLMClient

client = LLMClient()

# Tokens in Nachrichten zählen
messages = [
    {"role": "system", "content": "Du bist hilfsbereit."},
    {"role": "user", "content": "Erkläre KI im Detail."}
]
token_count = client.count_tokens(messages)
print(f"Dies wird ~{token_count} Tokens verwenden")

# Tokens in einem String zählen
text = "Hallo, wie geht es dir?"
tokens = client.count_string_tokens(text)
```

### 2. ⚡ Async-Unterstützung

```python
from llm_client import LLMClient

# Async-Client erstellen
async_client = LLMClient(use_async=True)

# Async Chat-Completion
response = await async_client.achat_completion(messages)

# Async Streaming
async for chunk in async_client.achat_completion_stream(messages):
    print(chunk, end="", flush=True)

# Async Tool-Calling
result = await async_client.achat_completion_with_tools(messages, tools)
```

### 3. 📁 Konfigurationsdateien

```python
from llm_client import LLMClient

# Client aus Konfigurationsdatei laden
client = LLMClient.from_config("llm_config.yaml")

# Spezifischen Provider aus Config verwenden
client = LLMClient.from_config("llm_config.yaml", provider="groq")
```

Beispiel `llm_config.yaml`:

```yaml
default_provider: openai

global_settings:
  temperature: 0.7
  max_tokens: 512

providers:
  openai:
    model: gpt-4o-mini
    temperature: 0.7

  groq:
    model: llama-3.3-70b-versatile
    temperature: 0.5

  gemini:
    model: gemini-2.0-flash-exp
    temperature: 0.8
```

### 4. ☁️ Ollama Cloud-Unterstützung

```python
from llm_client import LLMClient

# Automatische Cloud-Erkennung bei Modellen mit '-cloud' Suffix
client = LLMClient(llm="gpt-oss:120b-cloud")

# Oder explizit Cloud-Modus aktivieren
client = LLMClient(
    api_choice="ollama",
    llm="gpt-oss:120b-cloud",
    use_ollama_cloud=True
)

# Mit eigenem Ollama Cloud API Key
import os
os.environ["OLLAMA_API_KEY"] = "your-api-key"
client = LLMClient(llm="gpt-oss:120b-cloud")

# Nahtlos zwischen lokal und Cloud wechseln
client = LLMClient(api_choice="ollama", llm="llama3.2:1b")  # Lokal
client.switch_provider("ollama", llm="gpt-oss:120b-cloud", use_ollama_cloud=True)  # Cloud
```

**Verfügbare Cloud-Modelle:**
- `gpt-oss:120b-cloud` - GPT OSS 120B auf Ollama Cloud
- Weitere Modelle siehe [Ollama Cloud Dokumentation](https://ollama.com)

**Hybrid-Ansatz:**
```python
# Lokales Ollama für einfache Aufgaben (kostenlos, privat)
local_client = LLMClient(api_choice="ollama", llm="llama3.2:1b")
simple_response = local_client.chat_completion(simple_messages)

# Ollama Cloud für komplexe Aufgaben (leistungsstark)
cloud_client = LLMClient(llm="gpt-oss:120b-cloud")
complex_response = cloud_client.chat_completion(complex_messages)
```

**Vorteile von Ollama Cloud:**
- ✅ Zugriff auf leistungsstarke Modelle ohne lokale Hardware
- ✅ Schnellere Inferenz als lokale Ausführung
- ✅ Einfaches Umschalten zwischen lokal und Cloud
- ✅ Kompatibel mit allen bestehenden Features (Streaming, Async, etc.)

Siehe `examples/ollama_cloud_examples.py` für umfassende Beispiele.

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

### Mit allen Features

```bash
pip install -e ".[all]"
```

---

## 🚦 Schnellstart

```python
from llm_client import LLMClient

# Automatische API-Erkennung
client = LLMClient()

messages = [
    {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
    {"role": "user", "content": "Erkläre Machine Learning in einem Satz."}
]

response = client.chat_completion(messages)
print(response)
```

### Jupyter Notebook

Für einen umfassenden Überblick teste das Jupyter Notebook [llm_client_example.ipynb](notebooks/llm_client_example.ipynb) auf Google Colab.

---

## 🔧 Konfiguration

### API-Keys einrichten

Erstelle `secrets.env`:

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

## 📚 Verwendung

### 📊 Token-Zählung

Präzise Token-Zählung hilft, API-Kosten und Kontext-Limits zu verwalten:

```python
from llm_client import LLMClient

client = LLMClient()

# Tokens in Nachrichten zählen
messages = [
    {"role": "system", "content": "Du bist hilfsbereit."},
    {"role": "user", "content": "Was ist Quantencomputing?"}
]

token_count = client.count_tokens(messages)
print(f"Nachrichten enthalten {token_count} Tokens")

# Prüfen, ob im Budget
max_tokens = 4096
reserved_for_response = 500
available = max_tokens - token_count - reserved_for_response

if available > 0:
    print(f"✓ {available} Tokens verfügbar für Antwort")
else:
    print("✗ Nachricht zu lang!")

# Tokens in Plain-Text zählen
text = "Hallo, wie geht es dir heute?"
string_tokens = client.count_string_tokens(text)
print(f"String hat {string_tokens} Tokens")
```

**Features:**
- Nutzt tiktoken für präzise Zählung
- Unterstützt alle GPT-Modelle (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
- Fallback auf Schätzung wenn tiktoken nicht verfügbar
- Funktioniert mit jedem Provider

---

### ⚡ Async-Unterstützung

Vollständige async/await-Unterstützung für nicht-blockierende Operationen:

```python
from llm_client import LLMClient
import asyncio

async def main():
    # Async-Client erstellen
    client = LLMClient(use_async=True)

    messages = [{"role": "user", "content": "Was ist asynchrone Programmierung?"}]

    # Async Chat-Completion
    response = await client.achat_completion(messages)
    print(response)

    # Async Streaming
    print("\nStreaming-Antwort:")
    async for chunk in client.achat_completion_stream(messages):
        print(chunk, end="", flush=True)
    print()

    # Async Tool-Calling
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Wetter für einen Ort abrufen",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }]

    result = await client.achat_completion_with_tools(messages, tools)
    print(result)

# Async-Code ausführen
asyncio.run(main())
```

**Gleichzeitige Anfragen:**

```python
async def process_many_questions():
    client = LLMClient(use_async=True)

    questions = [
        "Was ist Python?",
        "Was ist JavaScript?",
        "Was ist Rust?"
    ]

    # Alle Fragen gleichzeitig verarbeiten
    tasks = [
        client.achat_completion([{"role": "user", "content": q}])
        for q in questions
    ]

    responses = await asyncio.gather(*tasks)

    for q, r in zip(questions, responses):
        print(f"F: {q}")
        print(f"A: {r[:100]}...\n")

asyncio.run(process_many_questions())
```

---

### 📁 Konfigurationsdateien

Verwalte mehrere Provider-Konfigurationen einfach:

**Config-Datei erstellen:**

```python
from llm_client.config import generate_config_template

# Template generieren
generate_config_template("llm_config.yaml", format="yaml")
```

**Beispiel-Konfiguration:**

```yaml
# llm_config.yaml
default_provider: openai

global_settings:
  temperature: 0.7
  max_tokens: 512

providers:
  openai:
    model: gpt-4o-mini
    temperature: 0.7
    max_tokens: 512

  groq:
    model: llama-3.3-70b-versatile
    temperature: 0.5
    max_tokens: 1024

  gemini:
    model: gemini-2.0-flash-exp
    temperature: 0.8
    max_tokens: 2048

  ollama:
    model: llama3.2:1b
    temperature: 0.7
    keep_alive: 5m
```

**Aus Config laden:**

```python
from llm_client import LLMClient

# Standard-Provider laden
client = LLMClient.from_config("llm_config.yaml")
print(f"Verwendet: {client.api_choice} - {client.llm}")

# Spezifischen Provider laden
groq_client = LLMClient.from_config("llm_config.yaml", provider="groq")
print(f"Verwendet: {groq_client.api_choice} - {groq_client.llm}")

# Async-Client aus Config laden
async_client = LLMClient.from_config("llm_config.yaml", use_async=True)
```

**Programmatische Konfiguration:**

```python
from llm_client.config import LLMConfig

config_dict = {
    "default_provider": "groq",
    "providers": {
        "groq": {
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.5
        }
    }
}

config = LLMConfig.from_dict(config_dict)

# Konfiguration validieren
is_valid, errors = config.validate()
if is_valid:
    print("✓ Konfiguration ist gültig")
else:
    print(f"✗ Fehler: {errors}")
```

---

### 🌊 Response-Streaming

Streame Antworten in Echtzeit für bessere Benutzererfahrung:

```python
from llm_client import LLMClient

client = LLMClient()
messages = [{"role": "user", "content": "Erzähle mir eine Geschichte über KI"}]

print("Streaming-Antwort:")
for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
print()
```

**Streaming mit Fehlerbehandlung:**

```python
from llm_client.exceptions import StreamingNotSupportedError, ChatCompletionError

try:
    for chunk in client.chat_completion_stream(messages):
        print(chunk, end="", flush=True)
except StreamingNotSupportedError:
    print("Streaming nicht unterstützt, verwende normale Completion")
    response = client.chat_completion(messages)
    print(response)
except ChatCompletionError as e:
    print(f"Fehler: {e}")
```

---

### 🔄 Dynamischer Provider-Wechsel

Wechsle zwischen Providern zur Laufzeit ohne neue Objekte zu erstellen:

```python
from llm_client import LLMClient

# Starte mit OpenAI
client = LLMClient(api_choice="openai", llm="gpt-4o-mini")
response1 = client.chat_completion([{"role": "user", "content": "Hallo"}])

# Wechsel zu Gemini
client.switch_provider("gemini", llm="gemini-2.0-flash-exp")
response2 = client.chat_completion([{"role": "user", "content": "Hallo"}])

# Wechsel zu Groq mit angepasster Temperatur
client.switch_provider("groq", temperature=0.3)
response3 = client.chat_completion([{"role": "user", "content": "Hallo"}])

# Wechsel zu lokalem Ollama
client.switch_provider("ollama")
response4 = client.chat_completion([{"role": "user", "content": "Hallo"}])
```

**Fallback-Strategie:**

```python
from llm_client import LLMClient
from llm_client.exceptions import ChatCompletionError

client = LLMClient(api_choice="openai")

try:
    response = client.chat_completion(messages)
except ChatCompletionError as e:
    print(f"OpenAI fehlgeschlagen: {e}")
    # Fallback zu Groq
    client.switch_provider("groq")
    response = client.chat_completion(messages)
```

**Kostenoptimierung:**

```python
client = LLMClient()

# Günstigeres Modell für einfache Aufgaben
client.switch_provider("groq", llm="llama-3.3-70b-versatile")
simple_response = client.chat_completion(simple_messages)

# Leistungsstärkeres Modell für komplexe Aufgaben
client.switch_provider("openai", llm="gpt-4o")
complex_response = client.chat_completion(complex_messages)
```

---

### 🧰 Tool-Calling (Function Calling)

Alle Provider unterstützen OpenAI-kompatibles Tool-Calling:

```python
from llm_client import LLMClient

client = LLMClient()

# Tools definieren
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Aktuelles Wetter an einem Ort abrufen",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Stadt und Land, z.B. Berlin, Deutschland"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }
}]

messages = [{"role": "user", "content": "Wie ist das Wetter in Berlin?"}]

# Tool-Calling-Anfrage stellen
result = client.chat_completion_with_tools(messages, tools)

# Prüfen, ob Tools aufgerufen wurden
if result['tool_calls']:
    for tool_call in result['tool_calls']:
        print(f"Aufrufen: {tool_call['function']['name']}")
        print(f"Argumente: {tool_call['function']['arguments']}")
else:
    print(f"Antwort: {result['content']}")
```

---

### 🔧 Erweiterte Verwendung

**Spezifisches Modell wählen:**

```python
client = LLMClient(
    llm="gpt-4o",
    temperature=0.5,
    max_tokens=2048
)
```

**API manuell wählen:**

```python
# Gemini erzwingen
client = LLMClient(api_choice="gemini", llm="gemini-2.5-pro")

# Ollama erzwingen (auch wenn API-Keys vorhanden)
client = LLMClient(api_choice="ollama")

# OpenAI explizit wählen
client = LLMClient(api_choice="openai", llm="gpt-4o")
```

**Gemini-Modelle nutzen:**

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

messages = [{"role": "user", "content": "Erkläre Quantencomputing"}]
response = client.chat_completion(messages)
print(response)
```

**Mit llama-index Integration:**

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
| Gemini | `gemini-2.0-flash-exp`             | Googles neuestes Modell (Dez 2024)  |
| Ollama | `llama3.2:1b`                      | Läuft lokal, kein API-Key nötig     |

### Verfügbare Gemini-Modelle

Basierend auf den aktuellen Google Gemini API Dokumenten (Dezember 2025):

**Stabile Modelle:**
- `gemini-2.5-pro` - Höchste Leistung für komplexe Aufgaben
- `gemini-2.5-flash` - Optimale Balance zwischen Geschwindigkeit und Intelligenz
- `gemini-2.5-flash-lite` - Für massive Skalierung optimiert
- `gemini-2.0-flash` - Kosteneffektives Allzweckmodell

**Experimentelle/Preview Modelle:**
- `gemini-3-pro` - Neuestes Modell mit erweitertem Reasoning (Preview)
- `gemini-2.0-flash-exp` - Experimentelles Flash-Modell

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

## 🏗️ Projekt-Architektur

Das Projekt verwendet ein **Strategy Pattern** mit klarer Trennung von Verantwortlichkeiten:

```
llm_client/
├── base_provider.py      # Abstract Base Class für alle Provider
├── providers.py          # Konkrete Provider-Implementierungen
│   ├── OpenAIProvider
│   ├── GroqProvider
│   ├── GeminiProvider
│   └── OllamaProvider
├── async_providers.py    # Async Provider-Implementierungen
│   ├── AsyncOpenAIProvider
│   ├── AsyncGroqProvider
│   └── AsyncGeminiProvider
├── provider_factory.py   # Factory für Provider-Erstellung
├── llm_client.py        # Hauptklasse (verwendet Strategy Pattern)
├── adapter.py           # llama-index Integration
├── token_counter.py     # Token-Zähl-Utilities
├── config.py            # Konfigurationsdatei-Unterstützung
└── exceptions.py        # Custom Exception-Klassen
```

### Design Principles

1. **Strategy Pattern**: Verschiedene LLM-APIs als austauschbare Strategien
2. **Factory Pattern**: Zentrale Provider-Erstellung und -Konfiguration
3. **Single Responsibility**: Jede Klasse hat eine klar definierte Aufgabe
4. **Dependency Injection**: Provider werden in LLMClient injiziert
5. **Extensibility**: Neue APIs können leicht hinzugefügt werden

### Provider hinzufügen

Um einen neuen Provider hinzuzufügen:

1. Implementiere `BaseProvider` in `providers.py`
2. Registriere den Provider in `ProviderFactory._provider_classes`
3. Füge Tests in `tests/test_llm_client.py` hinzu
4. Aktualisiere die Dokumentation

---

## 🧪 Tests ausführen

Siehe [TESTING.md](docs/TESTING.md).

---

## 📊 Vollständige Projektstruktur

```
llm_client/
├── .github/
│   └── workflows/               # CI/CD Pipelines
│       ├── tests.yml           # Automatisierte Tests
│       ├── lint.yml            # Code-Qualität
│       ├── codeql.yml          # Security Scanning
│       └── release.yml         # Release Automation
├── llm_client/
│   ├── __init__.py             # Package Exports
│   ├── base_provider.py        # Abstract Base Class
│   ├── providers.py            # Sync Provider-Implementierungen
│   ├── async_providers.py      # Async Provider-Implementierungen
│   ├── provider_factory.py     # Factory Pattern
│   ├── llm_client.py           # Hauptklasse
│   ├── adapter.py              # llama-index Integration
│   ├── token_counter.py        # Token-Zählung mit tiktoken
│   ├── config.py               # Konfigurationsdatei-Unterstützung
│   └── exceptions.py           # Custom Exception-Klassen
├── examples/
│   ├── streaming_example.py    # Streaming und Retry Beispiele
│   └── usage_examples.py       # Token-Zählung, Async, Config Beispiele
├── notebooks/
│   ├── llm_client_example.ipynb      # Demo-Notebook
│   ├── RAGChatbot_groq_API.ipynb     # RAG Tutorial
│   ├── utils.py                      # Hilfsfunktionen
│   └── README.md                     # Notebook-Dokumentation
├── tests/
│   ├── test_llm_client.py            # Haupttests
│   ├── test_switch_provider.py       # Provider-Wechsel Tests
│   ├── test_adapter.py               # Adapter-Tests
│   ├── test_base_provider.py         # Base Class Tests
│   ├── test_providers.py             # Provider Tests
│   ├── test_provider_factory.py      # Factory Tests
│   ├── test_new_features.py          # Streaming/Retry Tests
│   ├── tests_new_features_complete.py # Token/Async/Config Tests
│   ├── test_integration.py           # Integration Tests
│   └── README.md                     # Test-Dokumentation
├── main.py                           # Beispiel-Script
├── pyproject.toml                    # Dependencies & Config
├── requirements.txt                  # Pip Requirements
├── environment.yaml                  # Conda Environment
├── llm_config.yaml                   # Beispiel-Konfigurationsdatei
├── README.md                         # Diese Datei
├── CHANGELOG.md                      # Versionshistorie
├── CONTRIBUTING.md                   # Contribution Guidelines
└── LICENSE                           # MIT License
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

### Code-Stil

- **Formatierung**: Black (100 Zeichen pro Zeile)
- **Linting**: Ruff
- **Type Hints**: Vollständige Typ-Annotationen
- **Docstrings**: Google-Style
- **Tests**: Pytest mit >90% Coverage

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

## 📝 Changelog

Siehe [CHANGELOG.md](CHANGELOG.md).

---

## ⭐ Support

Wenn Ihnen dieses Projekt gefällt, geben Sie ihm einen Stern auf GitHub!

Fragen? Öffnen Sie ein [Issue](https://github.com/dgaida/llm_client/issues).
