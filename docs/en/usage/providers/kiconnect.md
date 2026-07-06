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
| `openai-gpt5.5` | Latest GPT model (default) |
| `openai-gpt5.4-mini` | Fast and efficient model |
| `openai-gpt-oss-120b` | Powerful open reasoning model |
| `mistralai-mistral-small-4-119b-2603` | Mistral Small model |
| `glm-4.7` | GLM model |
| `qwen3.5-397b-a17b` | Large Qwen model |
| `qwen3-omni-30b-a3b-instruct` | Qwen Omni model |
| `deepseek-r1-distill-llama-70b` | DeepSeek R1 model |
| `qwen-qwen3-embedding-8b` | Qwen Embedding model |

### Model Selection

```python
# Use default model (openai-gpt5.5)
client = LLMClient(api_choice="kiconnect")

# Select specific model
client = LLMClient(
    api_choice="kiconnect",
    llm="openai-gpt-oss-120b"
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
