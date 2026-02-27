# 🧠 LLM Client

A universal Python client for accessing various Large Language Models (LLMs) via **OpenAI**, [**Groq**](https://groq.com/), [**Google Gemini**](https://ai.google.dev/gemini-api), or [**Ollama**](https://ollama.com/) – with automatic API detection, dynamic provider switching, token counting, async support, and configuration file management.

---

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![codecov](https://codecov.io/gh/dgaida/llm_client/branch/master/graph/badge.svg)](https://codecov.io/gh/dgaida/llm_client)
[![Tests](https://github.com/dgaida/llm_client/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/llm_client/actions/workflows/tests.yml)
[![Code Quality](https://github.com/dgaida/llm_client/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/llm_client/actions/workflows/lint.yml)
[![CodeQL](https://github.com/dgaida/llm_client/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/llm_client/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## 📑 Table of Contents

- [Features](#-features)
- [What's New in v0.3.0](#-whats-new-in-v030)
- [Installation](#%EF%B8%8F-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [Token Counting](#-token-counting)
  - [Async Support](#-async-support)
  - [Configuration Files](#-configuration-files)
  - [Response Streaming](#-response-streaming)
  - [Dynamic Provider Switching](#-dynamic-provider-switching)
- [Supported APIs](#-supported-apis--default-models)
- [Documentation](#-documentation)
- [Tests](#-running-tests)
- [Architecture](#-project-architecture)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Features

### Core Features
* 🔍 **Automatic API Detection** - Uses available API keys or falls back to Ollama
* ⚙️ **Unified Interface** - One method for all LLM backends
* 🔄 **Dynamic Provider Switching** - Switch between APIs at runtime without creating new objects
* 🧩 **Flexible Configuration** - Model, temperature, tokens freely adjustable
* 🔐 **Google Colab Support** - Automatic loading of secrets from userdata
* 📦 **Zero-Config** - Works out-of-the-box with Ollama

### Architecture
* 🏗️ **Strategy Pattern** - Clean architecture with provider classes
* 🏭 **Factory Pattern** - Centralized provider creation and management
* 🧪 **Full Test Coverage** - pytest-based with >92% code coverage
* 🌟 **Google Gemini Support** - Via OpenAI compatibility mode

---

## ✨ What's New in v0.3.0

Version 0.3.0 introduces four major features:

### 1. 📊 Token Counting with tiktoken

```python
from llm_client import LLMClient

client = LLMClient()

# Count tokens in messages
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Explain AI in detail."}
]
token_count = client.count_tokens(messages)
print(f"This will use ~{token_count} tokens")

# Count tokens in a string
text = "Hello, how are you?"
tokens = client.count_string_tokens(text)
```

### 2. ⚡ Async Support

```python
from llm_client import LLMClient

# Create async client
async_client = LLMClient(use_async=True)

# Async chat completion
response = await async_client.achat_completion(messages)

# Async streaming
async for chunk in async_client.achat_completion_stream(messages):
    print(chunk, end="", flush=True)

# Async tool calling
result = await async_client.achat_completion_with_tools(messages, tools)
```

### 3. 📁 Configuration Files

```python
from llm_client import LLMClient

# Load client from config file
client = LLMClient.from_config("llm_config.yaml")

# Use specific provider from config
client = LLMClient.from_config("llm_config.yaml", provider="groq")
```

Example `llm_config.yaml`:

```yaml
default_provider: openai

global_settings:
  temperature: 0.7
  max_tokens: 512

providers:
  openai:
    model: gpt-4o-mini
    temperature: 0.7

  groq:
    model: llama-3.3-70b-versatile
    temperature: 0.5

  gemini:
    model: gemini-2.0-flash-exp
    temperature: 0.8
```

### 4. ☁️ Ollama Cloud Support

```python
from llm_client import LLMClient

# Automatic cloud detection for models with '-cloud' suffix
client = LLMClient(llm="gpt-oss:120b-cloud")

# Or explicitly enable cloud mode
client = LLMClient(
    api_choice="ollama",
    llm="gpt-oss:120b-cloud",
    use_ollama_cloud=True
)

# With your own Ollama Cloud API key
import os
os.environ["OLLAMA_API_KEY"] = "your-api-key"
client = LLMClient(llm="gpt-oss:120b-cloud")

# Seamlessly switch between local and cloud
client = LLMClient(api_choice="ollama", llm="llama3.2:1b")  # Local
client.switch_provider("ollama", llm="gpt-oss:120b-cloud", use_ollama_cloud=True)  # Cloud
```

**Available Cloud Models:**
- `gpt-oss:120b-cloud` - GPT OSS 120B on Ollama Cloud
- More models see [Ollama Cloud Documentation](https://ollama.com)

**Hybrid Approach:**
```python
# Local Ollama for simple tasks (free, private)
local_client = LLMClient(api_choice="ollama", llm="llama3.2:1b")
simple_response = local_client.chat_completion(simple_messages)

# Ollama Cloud for complex tasks (powerful)
cloud_client = LLMClient(llm="gpt-oss:120b-cloud")
complex_response = cloud_client.chat_completion(complex_messages)
```

**Benefits of Ollama Cloud:**
- ✅ Access powerful models without local hardware
- ✅ Faster inference than local execution
- ✅ Easy switching between local and cloud
- ✅ Compatible with all existing features (streaming, async, etc.)

See `examples/ollama_cloud_examples.py` for comprehensive examples.

---

## ⚙️ Installation

### Quick Install

```bash
pip install git+https://github.com/dgaida/llm_client.git
```

### Development Install

```bash
git clone https://github.com/dgaida/llm_client.git
cd llm_client
pip install -e ".[dev]"
```

### With llama-index Support

```bash
pip install -e ".[llama-index]"
```

### With All Features

```bash
pip install -e ".[all]"
```

---

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

### Jupyter Notebook

For a comprehensive overview, try the Jupyter notebook [llm_client_example.ipynb](notebooks/llm_client_example.ipynb) on Google Colab.

---

## 🔧 Configuration

### Setting up API Keys

Create `secrets.env`:

```bash
# OpenAI
OPENAI_API_KEY=sk-xxxxxxxx

# Or Groq
GROQ_API_KEY=gsk-xxxxxxxx

# Or Google Gemini
GEMINI_API_KEY=AIzaSy-xxxxxxxx
```

**Without API Keys**: Automatically uses local [Ollama](https://ollama.com/) (installation required).

### Google Colab

In Colab, keys are automatically loaded from `userdata`:

```python
# Add to Secrets → OPENAI_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY
from llm_client import LLMClient
client = LLMClient()  # Automatically loads from userdata
```

---

## 📚 Usage

### 📊 Token Counting

Accurate token counting helps manage API costs and context limits:

```python
from llm_client import LLMClient

client = LLMClient()

# Count tokens in messages
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is quantum computing?"}
]

token_count = client.count_tokens(messages)
print(f"Messages contain {token_count} tokens")

# Check if within budget
max_tokens = 4096
reserved_for_response = 500
available = max_tokens - token_count - reserved_for_response

if available > 0:
    print(f"✓ {available} tokens available for response")
else:
    print("✗ Message too long!")

# Count tokens in plain text
text = "Hello, how are you doing today?"
string_tokens = client.count_string_tokens(text)
print(f"String has {string_tokens} tokens")
```

**Features:**
- Uses tiktoken for accurate counting
- Supports all GPT models (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
- Falls back to estimation if tiktoken not available
- Works with any provider

---

### ⚡ Async Support

Full async/await support for non-blocking operations:

```python
from llm_client import LLMClient
import asyncio

async def main():
    # Create async client
    client = LLMClient(use_async=True)

    messages = [{"role": "user", "content": "What is async programming?"}]

    # Async chat completion
    response = await client.achat_completion(messages)
    print(response)

    # Async streaming
    print("\nStreaming response:")
    async for chunk in client.achat_completion_stream(messages):
        print(chunk, end="", flush=True)
    print()

    # Async tool calling
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }]

    result = await client.achat_completion_with_tools(messages, tools)
    print(result)

# Run async code
asyncio.run(main())
```

**Concurrent Requests:**

```python
async def process_many_questions():
    client = LLMClient(use_async=True)

    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?"
    ]

    # Process all questions concurrently
    tasks = [
        client.achat_completion([{"role": "user", "content": q}])
        for q in questions
    ]

    responses = await asyncio.gather(*tasks)

    for q, r in zip(questions, responses):
        print(f"Q: {q}")
        print(f"A: {r[:100]}...\n")

asyncio.run(process_many_questions())
```

---

### 📁 Configuration Files

Manage multiple provider configurations easily:

**Creating a Config File:**

```python
from llm_client import generate_config_template

# Generate template
generate_config_template("llm_config.yaml", format="yaml")
```

**Example Configuration:**

```yaml
# llm_config.yaml
default_provider: openai

global_settings:
  temperature: 0.7
  max_tokens: 512

providers:
  openai:
    model: gpt-4o-mini
    temperature: 0.7
    max_tokens: 512

  groq:
    model: llama-3.3-70b-versatile
    temperature: 0.5
    max_tokens: 1024

  gemini:
    model: gemini-2.0-flash-exp
    temperature: 0.8
    max_tokens: 2048

  ollama:
    model: llama3.2:1b
    temperature: 0.7
    keep_alive: 5m
```

**Loading from Config:**

```python
from llm_client import LLMClient

# Load default provider
client = LLMClient.from_config("llm_config.yaml")
print(f"Using: {client.api_choice} - {client.llm}")

# Load specific provider
groq_client = LLMClient.from_config("llm_config.yaml", provider="groq")
print(f"Using: {groq_client.api_choice} - {groq_client.llm}")

# Load async client from config
async_client = LLMClient.from_config("llm_config.yaml", use_async=True)
```

**Programmatic Configuration:**

```python
from llm_client import LLMConfig

config_dict = {
    "default_provider": "groq",
    "providers": {
        "groq": {
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.5
        }
    }
}

config = LLMConfig.from_dict(config_dict)

# Validate configuration
is_valid, errors = config.validate()
if is_valid:
    print("✓ Configuration is valid")
else:
    print(f"✗ Errors: {errors}")
```

---

### 🌊 Response Streaming

Stream responses in real-time for better user experience:

```python
from llm_client import LLMClient

client = LLMClient()
messages = [{"role": "user", "content": "Tell me a story about AI"}]

print("Streaming response:")
for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
print()
```

**Streaming with Error Handling:**

```python
from llm_client import StreamingNotSupportedError, ChatCompletionError

try:
    for chunk in client.chat_completion_stream(messages):
        print(chunk, end="", flush=True)
except StreamingNotSupportedError:
    print("Streaming not supported, using regular completion")
    response = client.chat_completion(messages)
    print(response)
except ChatCompletionError as e:
    print(f"Error: {e}")
```

---

### 🔄 Dynamic Provider Switching

Switch between providers at runtime without creating new objects:

```python
from llm_client import LLMClient

# Start with OpenAI
client = LLMClient(api_choice="openai", llm="gpt-4o-mini")
response1 = client.chat_completion([{"role": "user", "content": "Hello"}])

# Switch to Gemini
client.switch_provider("gemini", llm="gemini-2.0-flash-exp")
response2 = client.chat_completion([{"role": "user", "content": "Hello"}])

# Switch to Groq with adjusted temperature
client.switch_provider("groq", temperature=0.3)
response3 = client.chat_completion([{"role": "user", "content": "Hello"}])

# Switch to local Ollama
client.switch_provider("ollama")
response4 = client.chat_completion([{"role": "user", "content": "Hello"}])
```

**Fallback Strategy:**

```python
from llm_client import LLMClient
from llm_client import ChatCompletionError

client = LLMClient(api_choice="openai")

try:
    response = client.chat_completion(messages)
except ChatCompletionError as e:
    print(f"OpenAI failed: {e}")
    # Fallback to Groq
    client.switch_provider("groq")
    response = client.chat_completion(messages)
```

**Cost Optimization:**

```python
client = LLMClient()

# Use cheaper model for simple tasks
client.switch_provider("groq", llm="llama-3.3-70b-versatile")
simple_response = client.chat_completion(simple_messages)

# Use more capable model for complex tasks
client.switch_provider("openai", llm="gpt-4o")
complex_response = client.chat_completion(complex_messages)
```

---

### 📝 Logging

The LLM Client includes comprehensive logging to help with debugging and monitoring:

```python
from llm_client import LLMClient, setup_logging

# Enable INFO level logging
setup_logging(level="INFO")

client = LLMClient()
messages = [{"role": "user", "content": "Hello"}]
response = client.chat_completion(messages)
```

**Log Levels:**

- `DEBUG`: Maximum verbosity - shows all operations, API calls, token counts
- `INFO`: Moderate verbosity - shows provider initialization, switching, and high-level operations
- `WARNING`: Default level - shows only warnings and errors
- `ERROR`: Only errors
- `CRITICAL`: Only critical errors

**Configuration Options:**

```python
# Via function call
setup_logging(level="DEBUG")

# Via environment variable
import os
os.environ["LLM_CLIENT_LOG_LEVEL"] = "INFO"
setup_logging()

# Custom format
setup_logging(
    level="INFO",
    format_string="%(levelname)s - %(message)s"
)

# Disable logging
from llm_client import disable_logging
disable_logging()

# Re-enable logging
from llm_client import enable_logging
enable_logging("INFO")
```

**Example Output (INFO level):**

```
2024-12-08 10:30:15 - llm_client.llm_client - INFO - Creating provider for API: auto-detect
2024-12-08 10:30:15 - llm_client.provider_factory - INFO - Auto-selected API: openai
2024-12-08 10:30:15 - llm_client.providers - INFO - OpenAI client initialized with model gpt-4o-mini
2024-12-08 10:30:15 - llm_client.llm_client - INFO - Initialized with provider: openai, model: gpt-4o-mini
```

**What Gets Logged:**

- Provider initialization and switching
- API key availability (without exposing keys)
- Model selection and configuration
- API calls and responses (size, not content)
- Token counting operations
- Errors and warnings with context

**Best Practices:**

- **Development**: Use `DEBUG` or `INFO` for visibility
- **Production**: Use `WARNING` or `ERROR` to minimize noise
- **Testing**: Use `disable_logging()` to keep test output clean
- **Debugging Issues**: Temporarily enable `DEBUG` level

See [LOGGING.md](docs/en/development/logging.md) for complete logging documentation and examples.

---

### 🧰 Tool Calling (Function Calling)

All providers support OpenAI-compatible tool calling:

```python
from llm_client import LLMClient

client = LLMClient()

# Define tools
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }
}]

messages = [{"role": "user", "content": "What's the weather in Boston?"}]

# Make tool-calling request
result = client.chat_completion_with_tools(messages, tools)

# Check if tools were called
if result['tool_calls']:
    for tool_call in result['tool_calls']:
        print(f"Calling: {tool_call['function']['name']}")
        print(f"Arguments: {tool_call['function']['arguments']}")
else:
    print(f"Response: {result['content']}")
```

---

### 📁 File Upload Support

Upload and analyze files (images, PDFs, videos) with your prompts:

```python
from llm_client import LLMClient

# Create client
client = LLMClient(api_choice="openai", llm="gpt-4o")

# Analyze an image
messages = [{"role": "user", "content": "What do you see in this image?"}]
response = client.chat_completion_with_files(
    messages,
    files=["vacation_photo.jpg"]
)

# Analyze a PDF document
messages = [{"role": "user", "content": "Summarize this research paper"}]
response = client.chat_completion_with_files(
    messages,
    files=["research_paper.pdf"]
)

# Multiple files at once
messages = [{"role": "user", "content": "Compare these images"}]
response = client.chat_completion_with_files(
    messages,
    files=["image1.jpg", "image2.png", "chart.pdf"]
)
```

**Supported File Types by Provider:**

| Provider | Images | PDFs | Videos | Audio |
|----------|--------|------|--------|-------|
| OpenAI   | ✅ (GPT-4o+) | ✅ (GPT-4o+) | ❌ | ❌ |
| Gemini   | ✅     | ✅   | ✅     | ✅    |
| Groq     | ✅ (vision models) | ❌ | ❌ | ❌ |
| Ollama   | ✅ (llava, bakllava) | ❌ | ❌ | ❌ |

**Image Formats:** PNG, JPEG, WEBP, GIF  
**Document Formats:** PDF  
**Video Formats:** MP4, MOV, AVI (Gemini only)  
**Audio Formats:** MP3, WAV (Gemini only)

**Async File Upload:**

```python
from llm_client import LLMClient

# Create async client
client = LLMClient(api_choice="gemini", use_async=True)

messages = [{"role": "user", "content": "Analyze this video"}]
response = await client.achat_completion_with_files(
    messages,
    files=["demo_video.mp4"]
)
```

**File Validation:**

```python
from llm_client import validate_file_for_provider

# Check if file is supported before uploading
is_valid, error = validate_file_for_provider("document.pdf", "openai")
if is_valid:
    print("File is supported!")
else:
    print(f"Error: {error}")
```

**Vision Models with Ollama:**

```python
# Use local vision model
client = LLMClient(api_choice="ollama", llm="llava:7b")

messages = [{"role": "user", "content": "Describe this image"}]
response = client.chat_completion_with_files(
    messages,
    files=["photo.jpg"]
)
```

See `examples/file_upload_examples.py` for comprehensive examples.

---

### 🔧 Advanced Usage

**Choose Specific Model:**

```python
client = LLMClient(
    llm="gpt-4o",
    temperature=0.5,
    max_tokens=2048
)
```

**Manually Select API:**

```python
# Force Gemini
client = LLMClient(api_choice="gemini", llm="gemini-2.5-pro")

# Force Ollama (even if API keys present)
client = LLMClient(api_choice="ollama")

# Explicitly choose OpenAI
client = LLMClient(api_choice="openai", llm="gpt-4o")
```

**Using Gemini Models:**

```python
from llm_client import LLMClient

# Automatic if GEMINI_API_KEY is set
client = LLMClient()

# Or explicit with specific model
client = LLMClient(
    api_choice="gemini",
    llm="gemini-2.5-flash",
    temperature=0.7
)

messages = [{"role": "user", "content": "Explain quantum computing"}]
response = client.chat_completion(messages)
print(response)
```

**With llama-index Integration:**

```python
from llm_client import LLMClientAdapter, LLMClient

# Create adapter (works with Gemini too)
llm_adapter = LLMClientAdapter(client=LLMClient(api_choice="gemini"))

# Use in llama-index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, llm=llm_adapter)
```

---

## 🧩 Supported APIs & Default Models

| API    | Default Model                      | Notes                             |
| ------ |------------------------------------|-----------------------------------|
| OpenAI | `gpt-4o-mini`                      | Fast, reliable                    |
| Groq   | `moonshotai/kimi-k2-instruct-0905` | Very efficient on GroqCloud       |
| Gemini | `gemini-2.0-flash-exp`             | Google's latest model (Dec 2024)  |
| Ollama | `llama3.2:1b`                      | Runs locally, no API key needed   |

### Available Gemini Models

Based on current Google Gemini API documentation (December 2025):

**Stable Models:**
- `gemini-2.5-pro` - Highest performance for complex tasks
- `gemini-2.5-flash` - Optimal balance of speed and intelligence
- `gemini-2.5-flash-lite` - Optimized for massive scale
- `gemini-2.0-flash` - Cost-effective general-purpose model

**Experimental/Preview Models:**
- `gemini-3-pro` - Latest model with extended reasoning (Preview)
- `gemini-2.0-flash-exp` - Experimental Flash model

### Ollama Installation

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download

# Download a model
ollama pull llama3.2:1b
```

---

## 🏗️ Project Architecture

The project uses a **Strategy Pattern** with clear separation of concerns:

```
llm_client/
├── base_provider.py      # Abstract Base Class for all providers
├── providers.py          # Concrete provider implementations
│   ├── OpenAIProvider
│   ├── GroqProvider
│   ├── GeminiProvider
│   └── OllamaProvider
├── async_providers.py    # Async provider implementations
│   ├── AsyncOpenAIProvider
│   ├── AsyncGroqProvider
│   └── AsyncGeminiProvider
├── provider_factory.py   # Factory for provider creation
├── llm_client.py        # Main class (uses Strategy Pattern)
├── adapter.py           # llama-index integration
├── token_counter.py     # Token counting utilities
├── config.py            # Configuration file support
└── exceptions.py        # Custom exception classes
```

### Design Principles

1. **Strategy Pattern**: Different LLM APIs as interchangeable strategies
2. **Factory Pattern**: Centralized provider creation and configuration
3. **Single Responsibility**: Each class has a clearly defined purpose
4. **Dependency Injection**: Providers are injected into LLMClient
5. **Extensibility**: New APIs can easily be added

### Adding a New Provider

To add a new provider:

1. Implement `BaseProvider` in `providers.py`
2. Register the provider in `ProviderFactory._provider_classes`
3. Add tests in `tests/test_llm_client.py`
4. Update documentation

---

## 📖 Documentation

### Getting Started
- [Installation & Setup](docs/en/installation.md)
- [Quick Start Guide](docs/en/getting-started.md)
- [API Reference](docs/en/api/llm_client.md)

### Features
- [Token Counting](docs/en/usage/token-counting.md)
- [Async Support](docs/en/features.md#async-support)
- [Configuration Files](docs/en/features.md#configuration-files)
- [Response Streaming](docs/en/features.md#response-streaming)
- [Provider Switching](docs/en/features.md#dynamic-provider-switching)
- [Tool Calling](docs/en/features.md#tool-calling-function-calling)
- [File Upload](docs/en/features.md#file-upload)

### Provider Guides
- [OpenAI](docs/en/usage/providers/openai.md)
- [Groq](docs/en/usage/providers/groq.md)
- [Google Gemini](docs/en/usage/providers/gemini.md)
- [Ollama (local)](docs/en/usage/providers/ollama.md)
- [Ollama Cloud](docs/en/usage/providers/ollama_cloud.md)

### Other Resources
- [CLI Usage](docs/en/usage/cli.md)
- [Troubleshooting](docs/en/troubleshooting.md)
- [CHANGELOG](CHANGELOG.md)
- [Contributing Guidelines](CONTRIBUTING.md)

---

## 🧪 Running Tests

See [TESTING.md](docs/en/development/testing.md).

---

## 📊 Complete Project Structure

```
llm_client/
├── .github/
│   └── workflows/               # CI/CD Pipelines
│       ├── tests.yml           # Automated tests
│       ├── lint.yml            # Code quality
│       ├── codeql.yml          # Security scanning
│       └── release.yml         # Release automation
├── llm_client/
│   ├── __init__.py             # Package exports
│   ├── base_provider.py        # Abstract Base Class
│   ├── providers.py            # Sync provider implementations
│   ├── async_providers.py      # Async provider implementations
│   ├── provider_factory.py     # Factory Pattern
│   ├── llm_client.py           # Main class
│   ├── adapter.py              # llama-index integration
│   ├── token_counter.py        # Token counting with tiktoken
│   ├── config.py               # Configuration file support
│   └── exceptions.py           # Custom exception classes
├── examples/
│   ├── streaming_example.py    # Streaming and retry examples
│   └── usage_examples.py       # Token counting, async, config examples
├── notebooks/
│   ├── llm_client_example.ipynb      # Demo notebook
│   ├── RAGChatbot_groq_API.ipynb     # RAG tutorial
│   ├── utils.py                      # Helper functions
│   └── README.md                     # Notebook documentation
├── tests/
│   ├── test_llm_client.py            # Main tests
│   ├── test_switch_provider.py       # Provider switching tests
│   ├── test_adapter.py               # Adapter tests
│   ├── test_base_provider.py         # Base class tests
│   ├── test_providers.py             # Provider tests
│   ├── test_provider_factory.py      # Factory tests
│   ├── test_new_features.py          # Streaming/retry tests
│   ├── tests_new_features_complete.py # Token/async/config tests
│   ├── test_integration.py           # Integration tests
│   └── README.md                     # Test documentation
├── main.py                           # Example script
├── pyproject.toml                    # Dependencies & config
├── requirements.txt                  # Pip requirements
├── environment.yaml                  # Conda environment
├── llm_config.yaml                   # Example config file
├── README.md                         # This file
├── CHANGELOG.md                      # Version history
├── CONTRIBUTING.md                   # Contribution guidelines
└── LICENSE                           # MIT License
```

---

## 👥 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Developer Workflow

1. Fork & Clone
2. Create feature branch: `git checkout -b feature/my-feature`
3. Write and run tests
4. Format code: `black . && ruff check --fix .`
5. Commit & Push
6. Open Pull Request

### Code Style

- **Formatting**: Black (100 characters per line)
- **Linting**: Ruff
- **Type Hints**: Complete type annotations
- **Docstrings**: Google-style
- **Tests**: pytest with >90% coverage

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

© 2025 Daniel Gaida, Cologne University of Applied Sciences

---

## 🔗 Related Links

* [Ollama Documentation](https://github.com/ollama/ollama)
* [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
* [Groq Cloud](https://groq.com/)
* [Google Gemini API](https://ai.google.dev/gemini-api/docs)
* [Gemini OpenAI Compatibility](https://ai.google.dev/gemini-api/docs/openai)
* [llama-index Docs](https://docs.llamaindex.ai/)

---

## 📝 Version History

See [CHANGELOG.md](CHANGELOG.md).

---

## ⭐ Support

If you like this project, give it a star on GitHub!

Questions? Open an [Issue](https://github.com/dgaida/llm_client/issues).

---

## 🙏 Acknowledgments

This project was inspired by:
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [llama-index](https://github.com/run-llama/llama_index)

Special thanks to all contributors and users of this library!
