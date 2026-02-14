<p align="center">
  <img src="../../assets/logo4.png" width="250" alt="LLM Client Logo">
</p>

![Infografik](../../assets/infografik.png)

The **LLM Client** is a versatile Python tool that provides a **unified interface** for accessing various AI providers such as [**OpenAI**](https://openai.com/de-DE/api/), [**Groq**](https://groq.com/), [**Google Gemini**] (https://ai.google.dev/gemini-api), and [**Ollama**](https://ollama.com/). The software features **automatic API detection**, which flexibly falls back to a local Ollama instance if keys are missing. Technical highlights include precise **token counting**, full **async support**, and the ability to switch dynamically between different providers during runtime. Thanks to a clean architecture based on design patterns, the library also enables advanced **tool calling** and the upload of a wide variety of file formats. The library is **easy to use** compared to more complex frameworks and offers seamless integration into environments such as Google Colab.

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

---

## 🚀 Features

### Core Features
* 🔍 **Automatic API Detection** - Uses available API keys or falls back to Ollama
* ⚙️ **Unified Interface** - One method for all LLM backends
* 🔄 **Dynamic Provider Switching** - Switch between APIs at runtime without creating a new object
* 🧩 **Flexible Configuration** - Model, temperature, tokens freely adjustable
* 🔐 **Google Colab Support** - Automatic loading of secrets from userdata
* 📦 **Zero-Config** - Works out-of-the-box with Ollama
* 📊 **Token counting with tiktoken** - Precise token counting for cost management
* ⚡ **Full async support** - Async/await for all providers
* 📁 **Configuration files** - YAML/JSON configuration for multi-provider setups

### Architecture
* 🏗️ **Strategy Pattern** - Clean architecture with provider classes
* 🏭 **Factory Pattern** - Central provider creation and management

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
