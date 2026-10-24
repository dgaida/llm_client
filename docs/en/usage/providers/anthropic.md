# Anthropic (Claude) Provider Guide

The **Anthropic** provider allows using the Claude model family (e.g., `claude-3-5-sonnet-20241022`, `claude-3-7-sonnet-20250219`, `claude-3-5-haiku-20241022`, `claude-3-opus-20240229`) via the official Messages API.

## Prerequisites

Install the Anthropic extension or complete package:

```bash
pip install anthropic
# or
pip install -e ".[all]"
```

Set your API key in environment variables or in `secrets.env`:

```env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

## Quick Usage

```python
from llm_client import LLMClient

# Use the provider explicitly
client = LLMClient(api_choice="anthropic", llm="claude-3-5-sonnet-20241022")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in two sentences."}
]

response = client.chat_completion(messages)
print(response)
```

## Supported Features

- **Streaming**: Full support via `chat_completion_stream` / `achat_completion_stream`.
- **Tool Calling**: Automatic conversion of OpenAI tool definitions into Anthropic's tool schema.
- **File Upload / Multi-Modality**: Image (PNG, JPEG, WEBP, GIF) and PDF document support using content blocks.
- **Async API**: Available via `AsyncAnthropicProvider` with `LLMClient(api_choice="anthropic", use_async=True)`.
