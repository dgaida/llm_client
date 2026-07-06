# KI Connect Provider

The KI Connect provider enables access to AI models from TH Köln and other universities in NRW via the central infrastructure of [KI:connect.nrw](https://kiconnect.pages.rwth-aachen.de/pages/).

## Setup

### 1. Create API Key

1. Log in with your campusID at [ki.th-koeln.de](https://ki.th-koeln.de).  
2. Click on your name in the bottom left and select **"API Key Management"** (API-Schlüsselverwaltung).  
3. Create a new key and copy it.  

### 2. Configure

You can set the API key as an environment variable:

```bash
# Recommendation: Use KICONNECT_API_KEY
KICONNECT_API_KEY=your-api-key-here
```

## Usage

### Basic Usage

```python
from llm_client import LLMClient

# Explicit selection
client = LLMClient(api_choice="kiconnect")

messages = [{"role": "user", "content": "Hello!"}]
response = client.chat_completion(messages)
print(response)
```

### Available Models

KI Connect provides access to various models. You can find the names in the model overview in THKI Chat.

| Model | Description |
|-------|-------------|
| `GPT 5.4 mini` | Fast and efficient model (default) |
| `GPT OSS 120B` | Capable open source model (reasoning) |
| `Mistral Small 4 119B` | Open source model |
| `Qwen 3.5 397B` | Large open source model |

### Model Selection

```python
# Use default model (GPT 5.4 mini)
client = LLMClient(api_choice="kiconnect")

# Select specific model
client = LLMClient(
    api_choice="kiconnect",
    llm="GPT OSS 120B"
)
```

## Features

### Streaming

KI Connect supports streaming for real-time responses:

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
    response = await client.achat_completion([{"role": "user", "content": "Hello"}])
    print(response)

asyncio.run(main())
```

## Characteristics

- **Data Privacy:** KI Connect is GDPR-compliant and optimized for use in higher education in NRW.  
- **Infrastructure:** Models are hosted on university servers in NRW or via privacy-compliant connectors (GWDG/Azure).  
- **Interface:** The API is OpenAI-compatible.  

## Troubleshooting

### API Key Not Found

Ensure `KICONNECT_API_KEY` is set correctly or pass it directly:

```python
client = LLMClient(api_choice="kiconnect", kiconnect_api_key="your-key")
```

## Resources

- [THKI Chat (TH Köln)](https://ki.th-koeln.de)  
- [Information about THKI (Lehrpfade)](https://lehrpfade.th-koeln.de/thki-chat/)  
- [KI:connect.nrw Documentation](https://kiconnect.pages.rwth-aachen.de/pages/)  
