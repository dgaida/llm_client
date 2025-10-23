# 🧠 LLM Client

Ein universeller Python-Client zur Nutzung verschiedener Large Language Models (LLMs)
über **OpenAI**, **Groq** oder **Ollama** – mit automatischer API-Erkennung anhand von `secrets.env`.

---

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![Tests](https://github.com/dgaida/llm_client/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/llm_client/actions/workflows/tests.yml)
[![Code Quality](https://github.com/dgaida/llm_client/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/llm_client/actions/workflows/lint.yml)
[![CodeQL](https://github.com/dgaida/llm_client/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/llm_client/actions/workflows/codeql.yml)

## 🚀 Features

* 🔍 **Automatische API-Erkennung**
  * Nutzt `OPENAI_API_KEY` oder `GROQ_API_KEY`, falls vorhanden.
  * Fällt automatisch auf **Ollama** zurück, wenn keine API-Keys gefunden werden.
* ⚙️ **Einheitliches Interface**
  * Eine Methode `chat_completion(messages)` für alle Backends.
* 🧩 **Flexible Konfiguration**
  * Modell, Temperatur, Token-Limit und API-Typ frei wählbar.
* 🧪 **Testabdeckung**
  * Pytest-basiertes Testsuite inklusive Mocking und Fehlerhandling.

---

## 🧱 Projektstruktur

```
llm-client/
│
├── llm_client/
│   ├── __init__.py
│   └── llm_client.py
│
├── tests/
│   ├── __init__.py
│   └── test_llm_client.py
│
├── main.py
├── secrets.env
├── pyproject.toml
├── environment.yml
├── README.md
├── LICENSE                        # MIT Lizenz
└── .gitignore                     # Git-Ignore-Regeln
```

---

## ⚙️ Installation

### 🧠 1. Umgebung erstellen

Mit Conda:

```bash
conda env create -f environment.yml
conda activate llm-client-env
```

oder mit Mamba:

```bash
mamba env create -f environment.yml
mamba activate llm-client-env
```

---

### 🪄 2. Installation im Entwicklungsmodus

```bash
pip install -e .[dev]
```

Dies installiert:

* `llm_client` (das eigentliche Package)
* `pytest`, `ruff`, `pytest-cov` (für Tests & Linting)

---

### 🔑 3. API-Keys konfigurieren

Lege eine Datei `secrets.env` im Projektverzeichnis an:

```bash
OPENAI_API_KEY=sk-xxxxxxxx
# GROQ_API_KEY=groq-xxxxxxxx
```

Wenn keine Keys gesetzt sind, verwendet der Client automatisch **Ollama**.
🔗 Stelle sicher, dass Ollama lokal installiert ist.

[Ollama Python Download](https://www.ollama.com/download)

---

## 💡 Verwendung

### Beispiel `main.py`

```python
from llm_client import LLMClient

def main():
    client = LLMClient()  # automatische API-Erkennung

    print(f"Verwendete API: {client.api_choice}")
    print(f"Verwendetes Modell: {client.llm}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Erkläre kurz, was neuronale Netze sind."}
    ]

    response = client.chat_completion(messages)
    print("\nAntwort:\n", response)

if __name__ == "__main__":
    main()
```

Ausführen mit:

```bash
python main.py
```

---

## 🧢 Tests ausführen

```bash
pytest
```

Mit Code Coverage:

```bash
pytest --cov=llm_client
```

---

## ⚡️ Beispiele für Konfiguration

### API explizit wählen

```python
client = LLMClient(api_choice="ollama")
```

### Modell und Temperatur ändern

```python
client = LLMClient(llm="gpt-4o", temperature=0.5)
```

### Tokens und Keep-Alive anpassen

```python
client = LLMClient(max_tokens=2048, keep_alive="10m")
```

---

## 🧩 Unterstützte APIs & Default-Modelle

| API    | Default-Modell                 | Bemerkung                       |
| ------ | ------------------------------ | ------------------------------- |
| OpenAI | `gpt-4o-mini`                  | Schnell, zuverlässig            |
| Groq   | `moonshotai/kimi-k2-instruct-0905` | Sehr effizient auf GroqCloud    |
| Ollama | `llama3.2:1b`                     | Läuft lokal, kein API-Key nötig |

---

## 👨‍💻 Entwickler:innen-Info

**Tests & Linting**

```bash
pytest
ruff check llm_client
```

**Version bump (manuell):**

```bash
# in pyproject.toml ändern:
version = "0.1.1"
```

---

## 📄 Lizenz

Dieses Projekt steht unter der **MIT-Lizenz**.
© 2025 – Daniel Gaida, Technische Hochschule Köln.

---

## 🌐 Hinweise

* [Ollama Python API Doku](https://github.com/ollama/ollama/tree/main/api)
* [OpenAI Python SDK](https://github.com/openai/openai-python)
* [Groq SDK](https://github.com/groq/groq-python)
