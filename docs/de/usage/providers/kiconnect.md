# KI Connect Provider

Der KI Connect Provider ermöglicht den Zugriff auf die KI-Modelle der TH Köln und anderer Hochschulen in NRW über die zentrale Infrastruktur von [KI:connect.nrw](https://kiconnect.pages.rwth-aachen.de/pages/).

## Einrichtung

### 1. API-Schlüssel erstellen

1. Melden Sie sich mit Ihrer campusID auf [ki.th-koeln.de](https://ki.th-koeln.de) an.  
2. Klicken Sie unten links auf Ihren Namen und wählen Sie **"API-Schlüsselverwaltung"**.  
3. Erstellen Sie einen neuen Schlüssel und kopieren Sie diesen.  

### 2. Konfigurieren

Sie können den API-Schlüssel als Umgebungsvariable setzen:

```bash
# Empfehlung: Verwenden Sie KICONNECT_API_KEY
KICONNECT_API_KEY=ihr-api-schlüssel-hier
```

## Nutzung

### Grundlegende Nutzung

```python
from llm_client import LLMClient

# Explizite Auswahl
client = LLMClient(api_choice="kiconnect")

messages = [{"role": "user", "content": "Hallo!"}]
response = client.chat_completion(messages)
print(response)
```

### Verfügbare Modelle

KI Connect bietet Zugriff auf verschiedene Modelle. Die Namen können Sie der Modellübersicht im THKI Chat entnehmen.

| Modell | Beschreibung |
|-------|-------------|
| `openai-gpt5.5` | Neuestes GPT Modell (Standard) |
| `openai-gpt5.4-mini` | Schnelles und effizientes Modell |
| `openai-gpt-oss-120b` | Leistungsstarkes offenes Modell (Reasoning) |
| `mistralai-mistral-small-4-119b-2603` | Mistral Small Modell |
| `glm-4.7` | GLM Modell |
| `qwen3.5-397b-a17b` | Großes Qwen Modell |
| `qwen3-omni-30b-a3b-instruct` | Qwen Omni Modell |
| `deepseek-r1-distill-llama-70b` | DeepSeek R1 Modell |
| `qwen-qwen3-embedding-8b` | Qwen Embedding Modell |

### Modellauswahl

```python
# Standardmodell verwenden (openai-gpt5.5)
client = LLMClient(api_choice="kiconnect")

# Spezifisches Modell wählen
client = LLMClient(
    api_choice="kiconnect",
    llm="openai-gpt-oss-120b"
)
```

## Features

### Streaming

KI Connect unterstützt Streaming für Echtzeit-Antworten:

```python
for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
```

### Async Support

```python
import asyncio
from llm_client import LLMClient

async def main():
    client = LLMClient(api_choice="kiconnect", use_async=True)
    response = await client.achat_completion([{"role": "user", "content": "Hallo"}])
    print(response)

asyncio.run(main())
```

## Besonderheiten

- **Datenschutz:** KI Connect ist datenschutzkonform und für den Einsatz in der Lehre an Hochschulen in NRW optimiert.  
- **Infrastruktur:** Die Modelle werden teilweise auf eigenen Servern in NRW (z.B. RAMSES in Köln) oder über datenschutzkonforme Anbindungen (GWDG/Azure) bereitgestellt.  
- **Schnittstelle:** Die API ist OpenAI-kompatibel.  

## Fehlerbehebung

### API-Schlüssel nicht gefunden

Stellen Sie sicher, dass `KICONNECT_API_KEY` korrekt gesetzt ist oder übergeben Sie ihn direkt:

```python
client = LLMClient(api_choice="kiconnect", kiconnect_api_key="your-key")
```

## Ressourcen

- [THKI Chat (TH Köln)](https://ki.th-koeln.de)  
- [Informationen zu THKI (Lehrpfade)](https://lehrpfade.th-koeln.de/thki-chat/)  
- [KI:connect.nrw Dokumentation](https://kiconnect.pages.rwth-aachen.de/pages/)  
