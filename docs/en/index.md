# 🧠 LLM Client

![Infographic](../../assets/infografik.png)

```mermaid
graph TD
    subgraph "One Code"
        CODE["client = LLMClient()<br/>response = client.chat_completion(messages)"]
    end

    subgraph "Four APIs"
        OPENAI[OpenAI]
        GROQ[Groq]
        GEMINI[Gemini]
        OLLAMA[Ollama<br/>Local/Cloud]
    end

    subgraph "Many Possibilities"
        SWITCH[🔄 Switch Provider]
        TOKENS[📊 Monitor Costs]
        ASYNC[⚡ Async/Await]
        STREAM[🌊 Streaming]
        FILES[📎 Send Files]
    end

    CODE --> OPENAI
    CODE --> GROQ
    CODE --> GEMINI
    CODE --> OLLAMA

    GEMINI -.-> ASYNC
    GEMINI -.-> SWITCH
    GEMINI -.-> TOKENS
    GEMINI -.-> STREAM
    GEMINI -.-> FILES

    classDef codeClass fill:#e1f5ff,stroke:#01579b,stroke-width:3px,color:#000
    classDef apiClass fill:#e8f5e green,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef featureClass fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#000

    class CODE codeClass
    class OPENAI,GROQ,GEMINI,OLLAMA apiClass
    class SWITCH,TOKENS,ASYNC,STREAM,FILES featureClass
```

A universal Python client for accessing various Large Language Models (LLMs) via **OpenAI**, [**Groq**](https://groq.com/), [**Google Gemini**](https://ai.google.dev/gemini-api) or [**Ollama**](https://ollama.com/) – with automatic API detection, dynamic provider switching, token counting, async support, and configuration file management.

---

## 🚀 Features

### Core Features
* 🔍 **Automatic API Detection** - Uses available API keys or falls back to Ollama
* ⚙️ **Unified Interface** - One method for all LLM backends
* 🔄 **Dynamic Provider Switching** - Switch between APIs at runtime without creating a new object
* 🧩 **Flexible Configuration** - Model, temperature, tokens freely adjustable
* 🔐 **Google Colab Support** - Automatic loading of secrets from userdata
* 📦 **Zero-Config** - Works out-of-the-box with Ollama

### Architecture
* 🏗️ **Strategy Pattern** - Clean architecture with provider classes
* 🏭 **Factory Pattern** - Central provider creation and management
* 🧪 **Full Tests** - Pytest-based with >92% code coverage
* 🌟 **Google Gemini Support** - Use via OpenAI compatibility mode

## 🚦 Quick Start

```python
from llm_client import LLMClient

# Automatic API detection
client = LLMClient()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain machine learning in one sentence."}
]

response = client.chat_completion(messages)
print(response)
```

---

## 📖 Documentation

### Getting Started
- [Installation](installation.md)
- [Quick Start Guide](getting-started.md)
- [API Reference](api/index.md)

### Features
- [Token Counting](usage/token-counting.md)
- [Configuration Files](features.md)

### Further Resources
- [CLI Usage](usage/cli.md)
- [Troubleshooting](troubleshooting.md)
- [Changelog](changelog.md)
- [Contributing](development/contributing.md)
