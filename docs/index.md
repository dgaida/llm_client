# LLM Client Documentation

Welcome to the LLM Client documentation!

A universal Python client for accessing various Large Language Models through OpenAI, Groq, Google Gemini, or Ollama – with automatic API detection, dynamic provider switching, token counting, async support, and configuration file management.

---

## 🎯 Quick Links

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Quick Start__

    ---

    Get started in minutes with our installation guide

    [:octicons-arrow-right-24: Installation](getting_started.md)

-   :material-book-open-variant:{ .lg .middle } __Features__

    ---

    Explore all features of LLM Client

    [:octicons-arrow-right-24: Features Overview](features_overview.md)

-   :material-code-braces:{ .lg .middle } __API Reference__

    ---

    Complete API documentation

    [:octicons-arrow-right-24: API Docs](api_reference.md)

-   :material-package-variant:{ .lg .middle } __Examples__

    ---

    Real-world examples and use cases

    [:octicons-arrow-right-24: Examples](examples/basic-usage.md)

</div>

---

## ✨ Features

### Core Features
* 🔍 **Automatic API Detection** - Uses available API keys or falls back to Ollama
* ⚙️ **Unified Interface** - One method for all LLM backends
* 🔄 **Dynamic Provider Switching** - Switch between APIs at runtime without creating new objects
* 🧩 **Flexible Configuration** - Model, temperature, tokens freely adjustable
* 🔐 **Google Colab Support** - Automatic loading of secrets from userdata
* 📦 **Zero-Config** - Works out-of-the-box with Ollama

### Advanced Features (v0.3.0)
* 📊 **Token Counting** - Accurate token counting with tiktoken
* ⚡ **Async Support** - Full async/await support for non-blocking operations
* 📁 **Configuration Files** - YAML/JSON config for multi-provider setups
* 🌊 **Response Streaming** - Stream tokens in real-time
* 🧰 **Tool Calling** - OpenAI-compatible function calling
* 📎 **File Upload** - Send images, PDFs, videos with messages
* ☁️ **Ollama Cloud** - Access cloud models without local GPU

### Architecture
* 🏗️ **Strategy Pattern** - Clean architecture with provider classes
* 🏭 **Factory Pattern** - Centralized provider creation and management
* 🧪 **Full Test Coverage** - pytest-based with >92% code coverage

---

## 📦 Installation

=== "Quick Install"

    ```bash
    pip install git+https://github.com/dgaida/llm_client.git
    ```

=== "Development Install"

    ```bash
    git clone https://github.com/dgaida/llm_client.git
    cd llm_client
    pip install -e ".[dev]"
    ```

=== "With All Features"

    ```bash
    pip install -e ".[all]"
    ```

---

## 🚀 Quick Start

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

## 🔧 Supported Providers

| Provider | Default Model | Context Window | Strengths |
|----------|---------------|----------------|-----------|
| **OpenAI** | `gpt-4o-mini` | 128K tokens | Reliable, high quality |
| **Groq** | `moonshotai/kimi-k2-instruct-0905` | 128K tokens | Ultra-fast inference |
| **Gemini** | `gemini-2.0-flash-exp` | 1M-2M tokens | Long context, multimodal |
| **Ollama** | `llama3.2:1b` | varies | Local, private, free |

[:octicons-arrow-right-24: Provider Comparison](providers/openai.md)

---

## 📚 Documentation Structure

### Getting Started
- [Installation & Setup](getting_started.md)
- [Basic Usage](examples/basic-usage.md)
- [Configuration](features/configuration.md)

### Features
- [Token Counting](features/token_counting.md) - Manage costs and limits
- [Async Support](features/async_support.md) - Non-blocking operations
- [Streaming](features/streaming.md) - Real-time responses
- [Provider Switching](features/provider_switching.md) - Runtime flexibility
- [Tool Calling](features/tool_calling.md) - Function calling support
- [File Upload](features/file_upload.md) - Multimodal inputs

### Provider Guides
- [OpenAI Setup](providers/openai.md)
- [Groq Setup](providers/groq.md)
- [Gemini Setup](providers/gemini.md)
- [Ollama (Local)](providers/ollama.md)
- [Ollama Cloud](providers/ollama_cloud.md)

### Reference
- [API Reference](api_reference.md) - Complete API documentation
- [CLI Reference](cli.md) - Command-line interface
- [Troubleshooting](troubleshooting.md) - Common issues

---

## 🎯 Use Cases

### Cost Optimization
```python
# Use cheaper model for simple tasks
client.switch_provider("groq", llm="llama-3.3-70b-versatile")
simple_response = client.chat_completion(simple_messages)

# Use powerful model for complex tasks
client.switch_provider("openai", llm="gpt-4o")
complex_response = client.chat_completion(complex_messages)
```

### Fallback Strategy
```python
from llm_client.exceptions import ChatCompletionError

try:
    response = client.chat_completion(messages)
except ChatCompletionError:
    # Fallback to different provider
    client.switch_provider("groq")
    response = client.chat_completion(messages)
```

### Token Management
```python
# Check token count before sending
token_count = client.count_tokens(messages)
if token_count < 4000:
    response = client.chat_completion(messages)
else:
    print("Message too long!")
```

---

## 🏗️ Architecture

```
┌─────────────────┐
│   LLMClient     │ ◄── Main Interface
│  (Strategy)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ProviderFactory  │ ◄── Factory Pattern
│   (Creator)     │
└────────┬────────┘
         │
         ├────► OpenAIProvider
         ├────► GroqProvider
         ├────► GeminiProvider
         └────► OllamaProvider
```

[:octicons-arrow-right-24: Architecture Details](getting_started.md)

---

## 🤝 Contributing

Contributions are welcome! See our [Contributing Guide](CONTRIBUTING.md).

1. Fork the repository
2. Create a feature branch
3. Write tests
4. Submit a pull request

[:octicons-arrow-right-24: Contributing Guidelines](CONTRIBUTING.md)

---

## 📝 Version History

See [CHANGELOG.md](CHANGELOG.md) for version history.

**Latest:** v0.3.0 (January 2025)
- Token counting with tiktoken
- Full async/await support
- YAML/JSON configuration files
- Ollama Cloud support

---

## ⭐ Support

If you find this project helpful, please give it a star on GitHub!

- 📖 [Documentation](getting_started.md)
- 🐛 [Report Issues](https://github.com/dgaida/llm_client/issues)
- 💬 [Discussions](https://github.com/dgaida/llm_client/discussions)
- 📧 [Contact](mailto:daniel.gaida@th-koeln.de)

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](https://github.com/dgaida/llm_client/blob/master/LICENSE) for details.

© 2025 Daniel Gaida, Technische Hochschule Köln
