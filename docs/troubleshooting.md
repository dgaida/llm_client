# Troubleshooting

Lösungen für häufige Probleme mit LLM Client.

## Inhaltsverzeichnis

- [Installation](#installation)
- [API-Keys](#api-keys)
- [Provider-Probleme](#provider-probleme)
- [Ollama](#ollama)
- [Streaming](#streaming)
- [Token-Zählung](#token-zählung)
- [Async-Probleme](#async-probleme)
- [Konfigurationsdateien](#konfigurationsdateien)
- [Datei-Upload](#datei-upload)
- [Performance](#performance)

---

## Installation

### ImportError: No module named 'llm_client'

**Problem**: Nach der Installation kann llm_client nicht importiert werden.

**Lösung**:
```bash
# Überprüfe, ob Installation erfolgreich war
pip list | grep llm-client

# Neu installieren
pip uninstall llm-client
pip install git+https://github.com/dgaida/llm_client.git

# Oder mit editable install
git clone https://github.com/dgaida/llm_client.git
cd llm_client
pip install -e .
```

### ModuleNotFoundError: No module named 'openai'/'groq'/'ollama'

**Problem**: Provider-Pakete fehlen.

**Lösung**:
```bash
# Für OpenAI/Gemini
pip install openai

# Für Groq
pip install groq

# Für Ollama
pip install ollama

# Oder alle auf einmal
pip install openai groq ollama
```

### ImportError: tiktoken required

**Problem**: tiktoken nicht installiert, aber für Token-Zählung benötigt.

**Lösung**:
```bash
pip install tiktoken

# Oder alle optionalen Abhängigkeiten
pip install -e ".[all]"
```

---

## API-Keys

### APIKeyNotFoundError

**Problem**: API-Key nicht gefunden.

```python
APIKeyNotFoundError: OPENAI_API_KEY not found for openai provider.
```

**Lösung 1 - Environment Variable**:
```bash
# Linux/macOS
export OPENAI_API_KEY=sk-...

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."

# Windows (CMD)
set OPENAI_API_KEY=sk-...
```

**Lösung 2 - secrets.env Datei**:
```bash
# Erstelle secrets.env im Projektverzeichnis
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk-...
GEMINI_API_KEY=AIzaSy-...
```

**Lösung 3 - Google Colab**:
```python
# In Colab Secrets hinzufügen (🔑 Icon in linker Leiste)
# Key-Name: OPENAI_API_KEY
# Value: sk-...

from llm_client import LLMClient
client = LLMClient()  # Lädt automatisch aus Colab Secrets
```

**Lösung 4 - Programmatisch**:
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

from llm_client import LLMClient
client = LLMClient()
```

### API-Key wird nicht erkannt

**Problem**: Key ist gesetzt, aber wird nicht erkannt.

**Überprüfung**:
```python
import os

# Key ausgeben (sicher in lokaler Umgebung)
print(os.getenv("OPENAI_API_KEY"))

# In Colab
from google.colab import userdata
print(userdata.get("OPENAI_API_KEY"))
```

**Mögliche Ursachen**:
- Leerzeichen im Key
- Falscher Key-Name (z.B. `OPENAI_KEY` statt `OPENAI_API_KEY`)
- Key im falschen Format

---

## Provider-Probleme

### InvalidProviderError

**Problem**: Ungültiger Provider-Name.

```python
InvalidProviderError: Invalid provider: opena1. Valid providers are: openai, groq, gemini, ollama
```

**Lösung**:
```python
# Korrekte Provider-Namen
client = LLMClient(api_choice="openai")   # ✓
client = LLMClient(api_choice="groq")     # ✓
client = LLMClient(api_choice="gemini")   # ✓
client = LLMClient(api_choice="ollama")   # ✓

# Liste verfügbare Provider
from llm_client import ProviderFactory
available = ProviderFactory.get_available_providers()
print(f"Verfügbar: {available}")
```

### ProviderNotAvailableError

**Problem**: Provider-Paket nicht installiert.

```python
ProviderNotAvailableError: groq provider not available. Install with: pip install groq
```

**Lösung**:
```bash
# Installiere fehlendes Paket
pip install groq

# Oder alle Provider-Pakete
pip install openai groq ollama
```

### Provider-Wechsel schlägt fehl

**Problem**: `switch_provider()` führt zu Fehler.

**Lösung**:
```python
from llm_client.exceptions import APIKeyNotFoundError

try:
    client.switch_provider("groq")
except APIKeyNotFoundError as e:
    print(f"API-Key fehlt: {e.key_name}")
    # Setze Key und versuche erneut
    import os
    os.environ["GROQ_API_KEY"] = "gsk-..."
    client.switch_provider("groq")
```

---

## Ollama

### Ollama not running

**Problem**: Verbindung zu lokalem Ollama schlägt fehl.

```python
ChatCompletionError: Connection refused
```

**Lösung**:
```bash
# Prüfe, ob Ollama läuft
ollama list

# Starte Ollama-Service
# macOS/Linux
ollama serve

# Systemd (Linux)
sudo systemctl start ollama
sudo systemctl status ollama
```

### Model not found

**Problem**: Modell nicht verfügbar.

```python
ChatCompletionError: model 'llama3.2:1b' not found
```

**Lösung**:
```bash
# Liste installierte Modelle
ollama list

# Lade fehlendes Modell
ollama pull llama3.2:1b

# Beliebte Modelle
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull mixtral:8x7b
```

### Out of Memory (Ollama lokal)

**Problem**: Modell ist zu groß für verfügbaren RAM.

**Lösung**:
```python
# Verwende kleineres Modell
client = LLMClient(api_choice="ollama", llm="llama3.2:1b")  # ~1.3GB

# Oder quantisiertes Modell
client = LLMClient(api_choice="ollama", llm="llama3.1:8b-q4_0")  # Kleiner
```

**Empfehlungen nach RAM**:
- < 4GB RAM: `llama3.2:1b`
- 4-8GB RAM: `llama3.2:3b`
- 8-16GB RAM: `llama3.1:8b`
- 16GB+ RAM: `llama3.1:70b` oder größer

### Ollama Cloud API Key fehlt

**Problem**: Ollama Cloud ohne API-Key.

```python
APIKeyNotFoundError: OLLAMA_API_KEY not found for ollama_cloud provider
```

**Lösung**:
```bash
# Setze Ollama Cloud API Key
export OLLAMA_API_KEY=your_api_key

# Oder in secrets.env
echo "OLLAMA_API_KEY=your_api_key" >> secrets.env
```

---

## Streaming

### StreamingNotSupportedError

**Problem**: Streaming wird nicht unterstützt.

**Lösung**:
```python
from llm_client.exceptions import StreamingNotSupportedError

try:
    for chunk in client.chat_completion_stream(messages):
        print(chunk, end="")
except StreamingNotSupportedError:
    # Fallback auf normale Completion
    response = client.chat_completion(messages)
    print(response)
```

### Streaming friert ein

**Problem**: Stream stoppt mitten in der Antwort.

**Mögliche Ursachen**:
- Netzwerkprobleme
- Timeout
- API-Limit erreicht

**Lösung**:
```python
import time

# Mit Timeout-Handling
chunks = []
try:
    for chunk in client.chat_completion_stream(messages):
        chunks.append(chunk)
        print(chunk, end="", flush=True)
except Exception as e:
    print(f"\nStreaming unterbrochen: {e}")
    if chunks:
        print(f"Teilweise Antwort: {''.join(chunks)}")
```

---

## Token-Zählung

### tiktoken not available

**Problem**: Token-Zählung ohne tiktoken.

**Lösung**:
```bash
# Installiere tiktoken
pip install tiktoken
```

**Workaround ohne tiktoken**:
```python
from llm_client import TokenCounter

# Nutzt automatisch Schätzung
counter = TokenCounter()
token_count = counter.count_tokens(messages)
print(f"Geschätzte Tokens: {token_count}")
```

### Falsche Token-Zahl

**Problem**: Token-Count weicht stark ab.

**Ursachen**:
- Falsches Modell für Encoding
- tiktoken nicht installiert (Schätzung)

**Lösung**:
```python
# Spezifiziere korrektes Modell
token_count = client.count_tokens(messages, model="gpt-4o")

# Für spezifische Modelle
from llm_client import TokenCounter
counter = TokenCounter()

# GPT-4o
count = counter.count_tokens(messages, model="gpt-4o")

# GPT-3.5
count = counter.count_tokens(messages, model="gpt-3.5-turbo")
```

---

## Async-Probleme

### RuntimeError: Async methods not supported

**Problem**: Async-Methoden auf Sync-Client.

```python
RuntimeError: AsyncOpenAIProvider does not support async methods
```

**Lösung**:
```python
# Erstelle Async-Client
client = LLMClient(use_async=True)

# Nutze async-Methoden
import asyncio

async def main():
    response = await client.achat_completion(messages)
    print(response)

asyncio.run(main())
```

### Event loop is already running

**Problem**: In Jupyter/Colab.

**Lösung**:
```python
# In Jupyter/Colab
import nest_asyncio
nest_asyncio.apply()

# Dann wie gewohnt
asyncio.run(main())
```

### Async-Provider nicht verfügbar

**Problem**: Import-Fehler bei async_providers.

**Lösung**:
```bash
# Installiere async-Abhängigkeiten
pip install asyncio

# Oder komplette Installation
pip install -e ".[all]"
```

---

## Konfigurationsdateien

### FileNotFoundError: Config file not found

**Problem**: Config-Datei nicht gefunden.

**Lösung**:
```python
from pathlib import Path

# Prüfe Pfad
config_path = Path("llm_config.yaml")
if not config_path.exists():
    print(f"Datei nicht gefunden: {config_path.absolute()}")

    # Erstelle Template
    from llm_client.config import generate_config_template
    generate_config_template("llm_config.yaml")
```

### ValueError: Invalid configuration

**Problem**: Konfiguration ungültig.

**Lösung**:
```python
from llm_client.config import LLMConfig

# Validiere Config
config = LLMConfig.from_file("llm_config.yaml")
is_valid, errors = config.validate()

if not is_valid:
    print("Fehler in Konfiguration:")
    for error in errors:
        print(f"  - {error}")
```

**Häufige Fehler**:
- Fehlender `model` Parameter
- Ungültiger `default_provider`
- YAML-Syntax-Fehler

### ImportError: pyyaml required

**Problem**: YAML-Datei, aber pyyaml fehlt.

**Lösung**:
```bash
pip install pyyaml
```

**Alternative - JSON verwenden**:
```python
# Nutze JSON statt YAML
generate_config_template("llm_config.json", format="json")
```

---

## Datei-Upload

### FileNotFoundError: File not found

**Problem**: Datei existiert nicht.

**Lösung**:
```python
from pathlib import Path

files = ["image.jpg", "document.pdf"]

# Prüfe Dateien
for file_path in files:
    if not Path(file_path).exists():
        print(f"Datei nicht gefunden: {file_path}")
        print(f"Aktuelles Verzeichnis: {Path.cwd()}")
```

### ValueError: Unsupported file type

**Problem**: Dateityp nicht unterstützt.

**Lösung**:
```python
from llm_client.file_utils import validate_file_for_provider

# Prüfe Datei für Provider
is_valid, error = validate_file_for_provider("video.mp4", "openai")

if not is_valid:
    print(f"Fehler: {error}")
    # OpenAI unterstützt keine Videos
    # Nutze stattdessen Gemini
    client = LLMClient(api_choice="gemini")
```

**Provider-Dateiunterstützung**:
- OpenAI: Bilder, PDFs
- Gemini: Bilder, PDFs, Videos, Audio
- Groq: Nur Bilder (Vision-Modelle)
- Ollama: Nur Bilder (Vision-Modelle)

### FileUploadNotSupportedError

**Problem**: Provider unterstützt keine Datei-Uploads.

**Lösung**:
```python
from llm_client.exceptions import FileUploadNotSupportedError

try:
    response = client.chat_completion_with_files(messages, files=["image.jpg"])
except FileUploadNotSupportedError as e:
    print(f"Datei-Upload nicht unterstützt: {e}")
    # Wechsel zu Provider mit File-Support
    client.switch_provider("gemini")
    response = client.chat_completion_with_files(messages, files=["image.jpg"])
```

---

## Performance

### Langsame Antworten

**Problem**: API-Aufrufe dauern sehr lange.

**Mögliche Ursachen & Lösungen**:

**1. Netzwerk-Probleme**:
```python
import time

start = time.time()
response = client.chat_completion(messages)
elapsed = time.time() - start

print(f"Dauer: {elapsed:.2f}s")

# Wenn > 10s: Netzwerkproblem oder API-Überlastung
```

**2. Zu viele Tokens**:
```python
# Prüfe Token-Count
token_count = client.count_tokens(messages)
print(f"Tokens: {token_count}")

# Reduziere Input
if token_count > 2000:
    # Kürze Nachrichten oder nutze Zusammenfassung
    pass
```

**3. Falsches Modell**:
```python
# Schnellere Modelle verwenden
client = LLMClient(api_choice="groq")  # Sehr schnell
# oder
client = LLMClient(api_choice="openai", llm="gpt-4o-mini")  # Schneller als gpt-4o
```

### Rate Limit Errors

**Problem**: Zu viele Anfragen.

```python
ChatCompletionError: Rate limit exceeded
```

**Lösung**:
```python
import time

# Einfaches Retry mit Delay
for attempt in range(3):
    try:
        response = client.chat_completion(messages)
        break
    except Exception as e:
        if "rate" in str(e).lower():
            wait_time = (attempt + 1) * 5
            print(f"Rate limit, warte {wait_time}s...")
            time.sleep(wait_time)
        else:
            raise

# Oder nutze eingebautes Retry (automatisch)
response = client.chat_completion(messages)  # Retry ist bereits eingebaut
```

### Memory Errors (lokal Ollama)

**Problem**: Out of Memory bei großen Modellen.

**Lösung**:
```python
# 1. Kleineres Modell verwenden
client = LLMClient(api_choice="ollama", llm="llama3.2:1b")

# 2. Modell nach Nutzung entladen
client = LLMClient(api_choice="ollama", keep_alive="0")

# 3. Quantisiertes Modell
client = LLMClient(api_choice="ollama", llm="llama3.1:8b-q4_0")

# 4. Ollama Cloud verwenden (keine lokale GPU nötig)
client = LLMClient(llm="gpt-oss:120b-cloud")
```

---

## Weitere Hilfe

### Debug-Logging aktivieren

```python
from llm_client import enable_logging

# Aktiviere Debug-Logging
enable_logging("DEBUG")

# Jetzt werden alle internen Schritte geloggt
client = LLMClient()
response = client.chat_completion(messages)
```

### Detaillierte Fehlerinfos

```python
from llm_client.exceptions import ChatCompletionError

try:
    response = client.chat_completion(messages)
except ChatCompletionError as e:
    print(f"Provider: {e.provider}")
    print(f"Original Error: {e.original_error}")
    print(f"Error Type: {type(e.original_error).__name__}")
```

### Issue auf GitHub erstellen

Wenn das Problem weiterhin besteht:

1. Besuche [GitHub Issues](https://github.com/dgaida/llm_client/issues)
2. Prüfe, ob das Problem bereits gemeldet wurde
3. Erstelle ein neues Issue mit:
   - Python-Version (`python --version`)
   - Betriebssystem
   - LLM Client Version
   - Verwendeter Provider
   - Minimales Reproduktionsbeispiel
   - Vollständiger Fehler-Traceback

### Support-Kanäle

- [GitHub Issues](https://github.com/dgaida/llm_client/issues) - Bug Reports & Feature Requests
- [GitHub Discussions](https://github.com/dgaida/llm_client/discussions) - Fragen & Diskussionen
- [Dokumentation](https://github.com/dgaida/llm_client/tree/master/docs) - Vollständige Dokumentation
