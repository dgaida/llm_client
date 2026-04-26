# Features-Übersicht

LLM Client bietet eine umfassende Palette an Funktionen für die Arbeit mit verschiedenen LLM-Anbietern. Diese Seite gibt Ihnen einen Überblick über alle verfügbaren Features.

## Inhaltsverzeichnis

- [Automatische API-Erkennung](#automatische-api-erkennung)
- [Token-Zählung](#token-counting)
- [Async-Unterstützung](#async-support)
- [Konfigurationsdateien](#konfigurationsdateien)  
- [Response-Streaming](#response-streaming)  
- [Dynamischer Provider-Wechsel](#dynamischer-provider-wechsel)  
- [Tool-Calling](#tool-calling-function-calling)  
- [Datei-Upload](#datei-upload)  

---

## Automatische API-Erkennung {: #automatic-api-detection }

Der LLM Client kann automatisch entscheiden, welcher Provider genutzt werden soll, basierend auf den verfügbaren Umgebungsvariablen.

### Unterstützung für generische API-Keys

Zusätzlich zu provider-spezifischen Keys (`OPENAI_API_KEY`, etc.) unterstützt der Client eine generische `API_KEY` Variable. Der Client analysiert das Präfix des Schlüssels, um den Provider zu bestimmen:

| Präfix | Erkannter Provider |
|--------|-------------------|
| `sk-`   | OpenAI            |
| `gsk-`  | Groq              |
| `gsk_`  | Groq              |
| `AIza`  | Google Gemini     |

### Beispiel

```python
import os
from llm_client import LLMClient

# Nur einen generischen Key setzen
os.environ["API_KEY"] = "sk-..."

# Client erkennt automatisch, dass es sich um OpenAI handelt
client = LLMClient()
print(client.api_choice) # "openai"
```

---

## Token-Zählung {: #token-counting }

Präzise Token-Zählung hilft, API-Kosten und Kontext-Limits zu verwalten.

### Grundlegende Verwendung

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

# Tokens in einem String zählen
text = "Hallo, wie geht es dir heute?"
string_tokens = client.count_string_tokens(text)
print(f"String hat {string_tokens} Tokens")
```

### Budget-Verwaltung

```python
# Prüfen, ob im Budget
max_tokens = 4096
reserved_for_response = 500
token_count = client.count_tokens(messages)
available = max_tokens - token_count - reserved_for_response

if available > 0:
    print(f"✓ {available} Tokens verfügbar für Antwort")
    response = client.chat_completion(messages)
else:
    print("✗ Nachricht zu lang!")
```

### Features

- Nutzt tiktoken für präzise Zählung  
- Unterstützt alle GPT-Modelle (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)  
- Fallback auf Schätzung wenn tiktoken nicht verfügbar  
- Funktioniert mit jedem Provider  

### Weitere Informationen

- [API-Referenz: count_tokens()](api/llm_client.md#llm_client.llm_client.LLMClient.count_tokens)  
- [API-Referenz: count_string_tokens()](api/llm_client.md#llm_client.llm_client.LLMClient.count_string_tokens)  

---

## Async-Unterstützung {: #async-support }

Vollständige async/await-Unterstützung für nicht-blockierende Operationen.

### Grundlegende Verwendung

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

asyncio.run(main())
```

### Gleichzeitige Anfragen

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

### Async Tool-Calling

```python
async def main():
    client = LLMClient(use_async=True)

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

    messages = [{"role": "user", "content": "Wie ist das Wetter in Berlin?"}]
    result = await client.achat_completion_with_tools(messages, tools)
    print(result)

asyncio.run(main())
```

### Verfügbare Async-Methoden

- `achat_completion()` - Async Chat-Completion  
- `achat_completion_stream()` - Async Streaming  
- `achat_completion_with_tools()` - Async Tool-Calling  
- `achat_completion_with_files()` - Async Datei-Upload  

### Weitere Informationen

- [API-Referenz: Async-Methoden](api/llm_client.md)  

---

## Konfigurationsdateien {: #configuration-files }

Verwalte mehrere Provider-Konfigurationen einfach via YAML oder JSON.

### Config-Datei erstellen

```python
from llm_client.config import generate_config_template

# Template generieren
generate_config_template("llm_config.yaml", format="yaml")
```

### Beispiel-Konfiguration

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
    model: gemini-3.1-flash-lite-preview
    temperature: 0.8
    max_tokens: 2048

  ollama:
    model: llama3.2:1b
    temperature: 0.7
    keep_alive: 5m
```

### Aus Config laden

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

### Programmatische Konfiguration

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

### Weitere Informationen

- [API-Referenz: LLMConfig](api/config.md)  
- [API-Referenz: from_config()](api/llm_client.md#llm_client.llm_client.LLMClient.from_config)  

---

## Response-Streaming {: #response-streaming }

Streame Antworten in Echtzeit für bessere Benutzererfahrung.

### Grundlegende Verwendung

```python
from llm_client import LLMClient

client = LLMClient()
messages = [{"role": "user", "content": "Erzähle mir eine Geschichte über KI"}]

print("Streaming-Antwort:")
for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
print()
```

### Streaming mit Fehlerbehandlung

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

### Async Streaming

```python
import asyncio

async def stream_response():
    client = LLMClient(use_async=True)
    messages = [{"role": "user", "content": "Erzähle eine Geschichte"}]

    async for chunk in client.achat_completion_stream(messages):
        print(chunk, end="", flush=True)
    print()

asyncio.run(stream_response())
```

### Provider-Unterstützung

| Provider | Streaming |
|----------|-----------|
| OpenAI   | ✅        |
| Groq     | ✅        |
| Gemini   | ✅        |
| Ollama   | ✅        |

### Weitere Informationen

- [API-Referenz: chat_completion_stream()](api/llm_client.md#llm_client.llm_client.LLMClient.chat_completion_stream)  

---

## Dynamischer Provider-Wechsel {: #dynamic-provider-switching }

Wechsle zwischen Providern zur Laufzeit ohne neue Objekte zu erstellen.

### Grundlegende Verwendung

```python
from llm_client import LLMClient

# Starte mit OpenAI
client = LLMClient(api_choice="openai", llm="gpt-4o-mini")
response1 = client.chat_completion([{"role": "user", "content": "Hallo"}])

# Wechsel zu Gemini
client.switch_provider("gemini", llm="gemini-3.1-flash-lite-preview")
response2 = client.chat_completion([{"role": "user", "content": "Hallo"}])

# Wechsel zu Groq mit angepasster Temperatur
client.switch_provider("groq", temperature=0.3)
response3 = client.chat_completion([{"role": "user", "content": "Hallo"}])
```

### Fallback-Strategie

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

### Kostenoptimierung

```python
client = LLMClient()

# Günstigeres Modell für einfache Aufgaben
client.switch_provider("groq", llm="llama-3.3-70b-versatile")
simple_response = client.chat_completion(simple_messages)

# Leistungsstärkeres Modell für komplexe Aufgaben
client.switch_provider("openai", llm="gpt-4o")
complex_response = client.chat_completion(complex_messages)
```

### Weitere Informationen

- [API-Referenz: switch_provider()](api/llm_client.md#llm_client.llm_client.LLMClient.switch_provider)  

---

## Tool-Calling (Function Calling) {: #tool-calling-function-calling }

Alle Provider unterstützen OpenAI-kompatibles Tool-Calling.

### Grundlegende Verwendung

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

### Tool-Choice kontrollieren

```python
# Automatisch (Standard) - LLM entscheidet
result = client.chat_completion_with_tools(messages, tools, tool_choice="auto")

# Keine Tools verwenden
result = client.chat_completion_with_tools(messages, tools, tool_choice="none")

# Spezifisches Tool erzwingen
result = client.chat_completion_with_tools(
    messages,
    tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}}
)
```

### Provider-Unterstützung

| Provider | Tool-Calling |
|----------|--------------|
| OpenAI   | ✅           |
| Groq     | ✅           |
| Gemini   | ✅           |
| Ollama   | ⚠️ Experimentell |

### Weitere Informationen

- [API-Referenz: chat_completion_with_tools()](api/llm_client.md#llm_client.llm_client.LLMClient.chat_completion_with_tools)  

---

## Datei-Upload {: #file-upload }

Sende Bilder, PDFs und andere Dateien mit Chat-Anfragen.

### Grundlegende Verwendung

```python
from llm_client import LLMClient

client = LLMClient()

# Einzelne Datei
messages = [{"role": "user", "content": "Was ist in diesem Bild?"}]
response = client.chat_completion_with_files(
    messages,
    files=["vacation_photo.jpg"]
)

# Mehrere Dateien
messages = [{"role": "user", "content": "Analysiere diese Dokumente"}]
response = client.chat_completion_with_files(
    messages,
    files=["report.pdf", "chart.png", "data.jpg"]
)
```

### Bildanalyse mit Gemini

```python
client = LLMClient(api_choice="gemini")

messages = [{"role": "user", "content": "Beschreibe dieses Bild im Detail"}]
response = client.chat_completion_with_files(
    messages,
    files=["complex_diagram.png"]
)
print(response)
```

### PDF-Analyse mit OpenAI

```python
client = LLMClient(api_choice="openai", llm="gpt-4o")

messages = [{"role": "user", "content": "Fasse dieses Dokument zusammen"}]
response = client.chat_completion_with_files(
    messages,
    files=["research_paper.pdf"]
)
print(response)
```

### Provider-Unterstützung

| Provider | Unterstützte Dateitypen |
|----------|------------------------|
| OpenAI   | Bilder (PNG, JPEG, WEBP, GIF), PDFs |
| Gemini   | Bilder, PDFs, Videos, Audio |
| Groq     | Bilder (nur Vision-Modelle) |
| Ollama   | Bilder (nur Vision-Modelle) |

### Async Datei-Upload

```python
import asyncio

async def analyze_files():
    client = LLMClient(use_async=True)

    messages = [{"role": "user", "content": "Analysiere diese Dateien"}]
    response = await client.achat_completion_with_files(
        messages,
        files=["image1.jpg", "document.pdf"]
    )
    print(response)

asyncio.run(analyze_files())
```

### Weitere Informationen

- [API-Referenz: chat_completion_with_files()](api/llm_client.md#llm_client.llm_client.LLMClient.chat_completion_with_files)  
- [Datei-Upload-Utilities](api/utils_file_utils.md)  
