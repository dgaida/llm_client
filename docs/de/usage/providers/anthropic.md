# Anthropic (Claude) Provider Guide

Der **Anthropic** Provider ermöglicht die Nutzung der Claude-Modellfamilie (z. B. `claude-3-5-sonnet-20241022`, `claude-3-7-sonnet-20250219`, `claude-3-5-haiku-20241022`, `claude-3-opus-20240229`) über die offizielle Messages API.

## Voraussetzungen

Installiere die Anthropic Erweiterung oder das Komplettpaket:

```bash
pip install anthropic
# oder
pip install -e ".[all]"
```

Richte deinen API-Schlüssel in der Umgebung oder in `secrets.env` ein:

```env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

## Schnelle Verwendung

```python
from llm_client import LLMClient

# Verwende den Provider explizit
client = LLMClient(api_choice="anthropic", llm="claude-3-5-sonnet-20241022")

messages = [
    {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
    {"role": "user", "content": "Erkläre Quantencomputing in zwei Sätzen."}
]

response = client.chat_completion(messages)
print(response)
```

## Unterstützte Features

- **Streaming**: Unterstützung von `chat_completion_stream` / `achat_completion_stream`.  
- **Tool-Calling**: Automatische Konvertierung von OpenAI-Tool-Formaten in das Anthropic Tool-Schema.  
- **Datei-Upload / Multi-Modalität**: Unterstützung von Bildern (PNG, JPEG, WEBP, GIF) und PDF-Dokumenten über Base64-Blöcke.  
- **Async API**: Nutzung von `AsyncAnthropicProvider` via `LLMClient(api_choice="anthropic", use_async=True)`.  
