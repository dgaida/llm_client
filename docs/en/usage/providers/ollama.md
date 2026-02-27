# Ollama Provider

The Ollama provider enables running open-source LLMs locally on your machine without API keys or internet connection.

## Setup

### 1. Install Ollama

**macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [https://ollama.ai/download](https://ollama.ai/download)

### 2. Pull Models

```bash
# Small, fast model (recommended for testing)
ollama pull llama3.2:1b

# Medium model (good balance)
ollama pull llama3.2:3b

# Larger, more capable models
ollama pull llama3.1:8b
ollama pull llama3.1:70b
ollama pull mixtral:8x7b
```

### 3. Verify Installation

```bash
# Check Ollama is running
ollama list

# Test a model
ollama run llama3.2:1b "Hello!"
```

## Usage

### Basic Usage

```python
from llm_client import LLMClient

# Ollama is used automatically if no API keys are found
client = LLMClient()

# Or explicitly specify Ollama
client = LLMClient(api_choice="ollama")
```

### Available Models

Popular models available through Ollama:

| Model | Size | Parameters | Best For |
|-------|------|------------|----------|
| `llama3.2:1b` | 1.3GB | 1B | Quick testing, low-resource devices |
| `llama3.2:3b` | 2.0GB | 3B | General tasks, good balance |
| `llama3.1:8b` | 4.7GB | 8B | Complex reasoning, code generation |
| `llama3.1:70b` | 40GB | 70B | Advanced tasks, highest quality |
| `mixtral:8x7b` | 26GB | 47B | Multilingual, strong performance |
| `codellama:7b` | 3.8GB | 7B | Code generation and analysis |
| `phi3:mini` | 2.3GB | 3.8B | Microsoft's efficient model |

Browse all models at [https://ollama.ai/library](https://ollama.ai/library)

### Model Selection

```python
# Use default model (llama3.2:1b)
client = LLMClient(api_choice="ollama")

# Specify model
client = LLMClient(
    api_choice="ollama",
    llm="llama3.1:8b"
)

# With custom parameters
client = LLMClient(
    api_choice="ollama",
    llm="llama3.2:3b",
    temperature=0.7,
    max_tokens=512,
    keep_alive="10m"  # Ollama-specific: keep model in memory
)
```

## Features

### Chat Completion

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain recursion in programming."}
]

response = client.chat_completion(messages)
print(response)
```

### Streaming

```python
messages = [
    {"role": "user", "content": "Write a short story about AI"}
]

print("Response: ", end="")
for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
print()
```

### Keep-Alive Control

Ollama loads models into memory and keeps them loaded for faster subsequent requests:

```python
# Keep model in memory for 10 minutes
client = LLMClient(
    api_choice="ollama",
    keep_alive="10m"
)

# Keep model loaded indefinitely
client = LLMClient(
    api_choice="ollama",
    keep_alive="-1"
)

# Unload immediately after use
client = LLMClient(
    api_choice="ollama",
    keep_alive="0"
)
```

### Multi-Turn Conversations

```python
conversation = [
    {"role": "system", "content": "You are a Python tutor."},
    {"role": "user", "content": "What is a list comprehension?"}
]

# First response
response1 = client.chat_completion(conversation)
print(f"Assistant: {response1}\n")

# Continue conversation
conversation.append({"role": "assistant", "content": response1})
conversation.append({"role": "user", "content": "Can you show an example?"})

response2 = client.chat_completion(conversation)
print(f"Assistant: {response2}")
```

## Configuration

### Via Config File

```yaml
# llm_config.yaml
providers:
  ollama:
    model: llama3.2:3b
    temperature: 0.7
    max_tokens: 512
    keep_alive: 5m
```

```python
client = LLMClient.from_config("llm_config.yaml", provider="ollama")
```

### Runtime Parameters

```python
client = LLMClient(
    api_choice="ollama",
    llm="llama3.1:8b",
    temperature=0.5,      # Lower = more focused
    max_tokens=1024,      # Maximum response length
    keep_alive="15m"      # Keep model loaded for 15 minutes
)
```

## Best Practices

### 1. Choose Model Based on Resources

```python
import psutil

# Check available RAM
available_ram = psutil.virtual_memory().available / (1024**3)  # GB

if available_ram < 4:
    model = "llama3.2:1b"  # Low memory
elif available_ram < 8:
    model = "llama3.2:3b"  # Medium memory
elif available_ram < 16:
    model = "llama3.1:8b"  # High memory
else:
    model = "llama3.1:70b"  # Very high memory

client = LLMClient(api_choice="ollama", llm=model)
```

### 2. Optimize Keep-Alive

```python
# For interactive sessions - keep model loaded
interactive_client = LLMClient(
    api_choice="ollama",
    keep_alive="30m"
)

# For batch processing - unload after each request
batch_client = LLMClient(
    api_choice="ollama",
    keep_alive="0"
)
```

### 3. Use Streaming for Better UX

```python
# Streaming provides immediate feedback
for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
```

### 4. Model Quantization

Ollama automatically uses quantized models for efficiency:

```python
# These are automatically quantized for performance
client = LLMClient(api_choice="ollama", llm="llama3.1:8b")
# Runs Q4_0 quantization by default (4-bit quantization)

# For higher quality (larger size):
client = LLMClient(api_choice="ollama", llm="llama3.1:8b-q8_0")
# 8-bit quantization - better quality, more memory
```

### 5. Fallback Strategy

```python
from llm_client import LLMClient
from llm_client.exceptions import ChatCompletionError

# Try cloud APIs first, fall back to Ollama
try:
    client = LLMClient(api_choice="groq")
    response = client.chat_completion(messages)
except ChatCompletionError:
    print("Cloud API unavailable, using local Ollama...")
    client = LLMClient(api_choice="ollama")
    response = client.chat_completion(messages)
```

## Advanced Features

### Temperature Control

```python
# Deterministic (good for facts)
factual = LLMClient(
    api_choice="ollama",
    llm="llama3.1:8b",
    temperature=0.1
)

# Creative (good for stories)
creative = LLMClient(
    api_choice="ollama",
    llm="llama3.1:8b",
    temperature=1.5
)
```

### System Prompts

```python
# Specialized assistant
client = LLMClient(api_choice="ollama")

messages = [
    {
        "role": "system",
        "content": "You are an expert Python developer. Provide clear, concise code examples."
    },
    {
        "role": "user",
        "content": "How do I read a CSV file?"
    }
]

response = client.chat_completion(messages)
```

### Code Generation

```python
# Use code-specialized models
client = LLMClient(
    api_choice="ollama",
    llm="codellama:7b"
)

messages = [
    {
        "role": "user",
        "content": "Write a Python function to calculate Fibonacci numbers"
    }
]

code = client.chat_completion(messages)
print(code)
```

## Performance Optimization

### 1. GPU Acceleration

Ollama automatically uses GPU if available (NVIDIA, AMD, or Apple Silicon).

**Check GPU usage:**
```bash
# NVIDIA
nvidia-smi

# macOS
# Activity Monitor → GPU tab
```

### 2. Concurrent Requests

```python
from concurrent.futures import ThreadPoolExecutor

client = LLMClient(api_choice="ollama", keep_alive="30m")

questions = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?"
]

def process_question(question):
    return client.chat_completion([{"role": "user", "content": question}])

# Process concurrently
with ThreadPoolExecutor(max_workers=3) as executor:
    responses = list(executor.map(process_question, questions))

for q, r in zip(questions, responses):
    print(f"Q: {q}\nA: {r}\n")
```

### 3. Model Preloading

```python
# Preload model at startup
import subprocess

subprocess.run(["ollama", "run", "llama3.2:3b", "Hello"],
               capture_output=True)

# Now the model is loaded and ready
client = LLMClient(api_choice="ollama", llm="llama3.2:3b")
```

## Troubleshooting

### Ollama Not Running

```bash
# Start Ollama service
ollama serve

# Or on macOS/Linux
sudo systemctl start ollama
```

### Model Not Found

```bash
# List installed models
ollama list

# Pull missing model
ollama pull llama3.2:1b
```

### Out of Memory

```python
# Try smaller model
client = LLMClient(api_choice="ollama", llm="llama3.2:1b")

# Or use quantized version
client = LLMClient(api_choice="ollama", llm="llama3.1:8b-q4_0")
```

### Slow Response Times

```python
# Ensure model stays loaded
client = LLMClient(
    api_choice="ollama",
    keep_alive="60m"  # Keep loaded for 1 hour
)

# Use smaller, faster model
client = LLMClient(api_choice="ollama", llm="llama3.2:1b")
```

## Error Handling

```python
from llm_client.exceptions import (
    ProviderNotAvailableError,
    ChatCompletionError
)

try:
    client = LLMClient(api_choice="ollama")
    response = client.chat_completion(messages)
except ProviderNotAvailableError:
    print("Ollama not installed!")
    print("Install from: https://ollama.ai")
except ChatCompletionError as e:
    print(f"Error: {e}")
    print("Is Ollama running? Try: ollama serve")
```

## Comparison with Cloud APIs

**Advantages:**
- ✅ Completely private - data never leaves your machine
- ✅ No API costs
- ✅ No rate limits
- ✅ Works offline
- ✅ Full control over models

**Considerations:**
- ⚠️ Requires local compute resources
- ⚠️ Slower than cloud APIs on typical hardware
- ⚠️ Model quality varies (generally lower than GPT-4)
- ⚠️ Need to manage model downloads

## Resources

- [Ollama Website](https://ollama.ai/)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Model Library](https://ollama.ai/library)
- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/README.md)

## Example: Complete Local RAG System

```python
from llm_client import LLMClient

# Setup local LLM
client = LLMClient(
    api_choice="ollama",
    llm="llama3.1:8b",
    keep_alive="30m"
)

# Load document (simplified example)
document = """
LLM Client is a universal Python client for accessing various
Large Language Models through OpenAI, Groq, Gemini, or Ollama.
It features automatic API detection, dynamic provider switching,
and a unified interface.
"""

# RAG-style query
query = "What is LLM Client?"

messages = [
    {
        "role": "system",
        "content": "Answer based on the following document:\n\n" + document
    },
    {
        "role": "user",
        "content": query
    }
]

# Get response with streaming
print("Answer: ", end="")
for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
print()
```

## Custom Ollama Parameters

Ollama supports additional parameters through the options field:

```python
# Direct ollama usage (advanced)
import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=messages,
    options={
        "temperature": 0.7,
        "num_predict": 512,  # max_tokens
        "top_k": 40,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    }
)
```

Note: LLMClient uses sensible defaults for these parameters. For fine-grained control, use the ollama package directly.
