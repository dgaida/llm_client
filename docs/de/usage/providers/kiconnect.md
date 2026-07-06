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
| `GPT 5.4 mini` | Schnelles und effizientes Modell (Standard) |
| `GPT OSS 120B` | Leistungsstarkes offenes Modell (Reasoning) |
| `Mistral Small 4 119B` | Offenes Modell |
| `Qwen 3.5 397B` | Großes offenes Modell |

### Modellauswahl

```python
# Standardmodell verwenden (GPT 5.4 mini)
client = LLMClient(api_choice="kiconnect")

# Spezifisches Modell wählen
client = LLMClient(
    api_choice="kiconnect",
    llm="GPT OSS 120B"
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
