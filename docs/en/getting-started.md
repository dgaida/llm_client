# Getting Started

This guide will help you get up and running with the LLM Client quickly.

## Basic Configuration

### 1. Configure API Keys

Create a `secrets.env` file in your project directory:

```bash
# OpenAI
OPENAI_API_KEY=sk-xxxxxxxx

# Groq (optional)
GROQ_API_KEY=gsk-xxxxxxxx

# Google Gemini (optional)
GEMINI_API_KEY=AIzaSy-xxxxxxxx
```

### 2. First Example

```python
from llm_client import LLMClient

# Automatic API detection
client = LLMClient()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
]

response = client.chat_completion(messages)
print(response)
```

## Key Concepts

### Automatic Provider Selection

The client automatically selects the first available API key:

1. **OpenAI** (if `OPENAI_API_KEY` is set)
2. **Groq** (if `GROQ_API_KEY` is set)
3. **Gemini** (if `GEMINI_API_KEY` is set)
4. **Ollama** (Fallback, requires local installation)

### Manual Selection

```python
# Force specific provider
client = LLMClient(api_choice="gemini")

# With custom model and parameters
client = LLMClient(
    api_choice="openai",
    llm="gpt-4o",
    temperature=0.5,
    max_tokens=2048
)
```

## Core Features

- **Chat Completion**: Standard requests to LLMs.
- **Streaming**: Get responses in real-time.
- **Token Counting**: Monitor your usage and costs.
- **Provider Switching**: Change the API at runtime.
- **Configuration Files**: Load settings from YAML or JSON.
- **Async Support**: Use `async/await` for high-performance applications.

## Next Steps

- [Configuration](configuration.md) - Configuration details
- [API Reference](api/index.md) - Detailed class documentation
- [Examples](usage/basic-usage.md) - Real-world use cases
