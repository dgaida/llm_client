# Ollama Cloud Support

The LLM Client now supports both local Ollama instances and Ollama Cloud for running large models without local GPU requirements.

## Overview

**Ollama Cloud** allows you to run large language models in the cloud without needing powerful local hardware. This is ideal for:
- Running models that don't fit on your local machine
- Working on devices without GPUs
- Quick prototyping without model downloads
- Accessing cloud-exclusive models

## Setup

### 1. Get Ollama Cloud API Key

First, create an account and get your API key:

```bash
# Sign in to Ollama Cloud
ollama signin

# Or create an API key at: https://ollama.com/settings/keys
```

### 2. Set Environment Variable

Add your API key to your environment:

```bash
# In .env or secrets.env file
OLLAMA_API_KEY=your_api_key_here

# Or export directly
export OLLAMA_API_KEY=your_api_key_here
```

## Usage

### Basic Cloud Usage

```python
from llm_client import LLMClient

# Method 1: Explicit cloud mode
client = LLMClient(
    api_choice="ollama",
    llm="gpt-oss:120b-cloud",
    use_ollama_cloud=True
)

# Method 2: Auto-detect from model name (ends with -cloud)
client = LLMClient(
    api_choice="ollama",
    llm="gpt-oss:120b-cloud"
)

# Method 3: From environment with cloud model
client = LLMClient(llm="gpt-oss:120b-cloud")

# Use it like any other provider
messages = [{"role": "user", "content": "Explain quantum computing"}]
response = client.chat_completion(messages)
print(response)
```

### Available Cloud Models

Cloud models are marked with `-cloud` suffix:

| Model | Size | Description |
|-------|------|-------------|
| `gpt-oss:120b-cloud` | 120B | Large open-source model via Ollama Cloud |

Check [ollama.com/search?c=cloud](https://ollama.com/search?c=cloud) for the latest cloud models.

### Streaming with Cloud

```python
from llm_client import LLMClient

client = LLMClient(llm="gpt-oss:120b-cloud")

messages = [{"role": "user", "content": "Tell me a story about AI"}]

print("Response: ", end="")
for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
print()
```

### Switch Between Local and Cloud

```python
from llm_client import LLMClient

# Start with local Ollama
client = LLMClient(api_choice="ollama", llm="llama3.2:1b")
local_response = client.chat_completion(messages)

# Switch to cloud for larger model
client.switch_provider(
    "ollama",
    llm="gpt-oss:120b-cloud",
    use_ollama_cloud=True
)
cloud_response = client.chat_completion(messages)

# Switch back to local
client.switch_provider(
    "ollama",
    llm="llama3.2:1b",
    use_ollama_cloud=False
)
```

### Configuration File

Add Ollama Cloud to your config file:

```yaml
# llm_config.yaml
providers:
  # Local Ollama
  ollama_local:
    model: llama3.2:1b
    temperature: 0.7
    use_cloud: false
    keep_alive: 5m

  # Ollama Cloud
  ollama_cloud:
    model: gpt-oss:120b-cloud
    temperature: 0.7
    use_cloud: true
```

Load and use:

```python
# Use local
client = LLMClient.from_config("llm_config.yaml", provider="ollama_local")

# Use cloud
client = LLMClient.from_config("llm_config.yaml", provider="ollama_cloud")
```

## Direct API Access (Advanced)

For direct access to Ollama Cloud API without local Ollama installation:

```python
from llm_client import LLMClient

# Direct cloud access with custom host
client = LLMClient(
    api_choice="ollama",
    llm="gpt-oss:120b",
    use_ollama_cloud=True,
    ollama_host="https://ollama.com"
)
```

## Comparison: Local vs Cloud

### Local Ollama

**Pros:**
- ✅ Complete privacy - data never leaves your machine
- ✅ No API costs
- ✅ No rate limits
- ✅ Works offline
- ✅ Full control

**Cons:**
- ⚠️ Requires local compute resources
- ⚠️ Limited by available hardware
- ⚠️ Need to manage model downloads

### Ollama Cloud

**Pros:**
- ✅ Access to large models (120B+)
- ✅ No local GPU required
- ✅ No model downloads
- ✅ Fast inference
- ✅ Works on any device

**Cons:**
- ⚠️ Requires API key
- ⚠️ Data sent to cloud
- ⚠️ Potential costs (check pricing)
- ⚠️ Requires internet connection

## Best Practices

### 1. Hybrid Approach

Use local for small tasks, cloud for complex ones:

```python
from llm_client import LLMClient

# Small tasks - use local
local_client = LLMClient(
    api_choice="ollama",
    llm="llama3.2:1b"
)

# Complex tasks - use cloud
cloud_client = LLMClient(
    api_choice="ollama",
    llm="gpt-oss:120b-cloud"
)

# Simple question - local
quick_answer = local_client.chat_completion([
    {"role": "user", "content": "What is Python?"}
])

# Complex analysis - cloud
detailed_answer = cloud_client.chat_completion([
    {"role": "user", "content": "Provide a detailed analysis of..."}
])
```

### 2. Fallback Strategy

Try local first, fallback to cloud:

```python
from llm_client import LLMClient
from llm_client.exceptions import ChatCompletionError

messages = [{"role": "user", "content": "Your query here"}]

try:
    # Try local first
    client = LLMClient(api_choice="ollama", llm="llama3.2:1b")
    response = client.chat_completion(messages)
except ChatCompletionError:
    # Fallback to cloud
    print("Local Ollama unavailable, using cloud...")
    client = LLMClient(llm="gpt-oss:120b-cloud")
    response = client.chat_completion(messages)

print(response)
```

### 3. Environment-Specific Configuration

```python
import os
from llm_client import LLMClient

# Use cloud in production, local in development
if os.getenv("ENVIRONMENT") == "production":
    client = LLMClient(llm="gpt-oss:120b-cloud")
else:
    client = LLMClient(api_choice="ollama", llm="llama3.2:1b")
```

## Python Library Examples

### Complete Example

```python
from llm_client import LLMClient

# Initialize cloud client
client = LLMClient(
    api_choice="ollama",
    llm="gpt-oss:120b-cloud",
    temperature=0.7,
    max_tokens=1024
)

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain machine learning in simple terms."}
]

response = client.chat_completion(messages)
print(f"Response: {response}")

# Streaming response
print("\nStreaming response:")
for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
print()

# Token counting
token_count = client.count_tokens(messages)
print(f"\nTokens used: {token_count}")
```

### Error Handling

```python
from llm_client import LLMClient
from llm_client.exceptions import (
    APIKeyNotFoundError,
    ChatCompletionError,
    ProviderNotAvailableError
)

try:
    client = LLMClient(
        api_choice="ollama",
        llm="gpt-oss:120b-cloud"
    )
    response = client.chat_completion(messages)

except APIKeyNotFoundError:
    print("Ollama Cloud API key not found!")
    print("Set OLLAMA_API_KEY environment variable or sign in with: ollama signin")

except ProviderNotAvailableError:
    print("Ollama package not installed!")
    print("Install with: pip install ollama")

except ChatCompletionError as e:
    print(f"Chat completion failed: {e}")
```

## Troubleshooting

### API Key Not Found

```python
# Error: APIKeyNotFoundError for ollama_cloud
# Solution: Set your API key
export OLLAMA_API_KEY=your_key_here
```

### Model Not Available

```python
# If cloud model not found, check available models:
# Visit: https://ollama.com/search?c=cloud

# Or list via API:
# curl https://ollama.com/api/tags
```

### Connection Issues

```python
# If connection fails, check:
# 1. Internet connection
# 2. API key is valid
# 3. Ollama cloud service is up
# 4. Try with custom host:

client = LLMClient(
    api_choice="ollama",
    llm="gpt-oss:120b-cloud",
    ollama_host="https://ollama.com"
)
```

## Migration from Local to Cloud

### Before (Local Only)

```python
client = LLMClient(
    api_choice="ollama",
    llm="llama3.2:1b"
)
```

### After (Cloud Enabled)

```python
# Option 1: Explicit cloud
client = LLMClient(
    api_choice="ollama",
    llm="gpt-oss:120b-cloud",
    use_ollama_cloud=True
)

# Option 2: Auto-detect from model name
client = LLMClient(
    api_choice="ollama",
    llm="gpt-oss:120b-cloud"
)
```

## Additional Resources

- [Ollama Cloud Documentation](https://docs.ollama.com/cloud)
- [Ollama Cloud Models](https://ollama.com/search?c=cloud)
- [Create API Keys](https://ollama.com/settings/keys)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [LLM Client Documentation](https://github.com/dgaida/llm_client)

## Example: RAG with Ollama Cloud

```python
from llm_client import LLMClient

# Use large cloud model for better RAG performance
client = LLMClient(llm="gpt-oss:120b-cloud")

# Your document
document = """
Ollama Cloud is a new service that allows running large language
models in the cloud without requiring powerful local hardware.
"""

# RAG query
query = "What is Ollama Cloud?"

messages = [
    {
        "role": "system",
        "content": f"Answer based on this document:\n\n{document}"
    },
    {
        "role": "user",
        "content": query
    }
]

response = client.chat_completion(messages)
print(f"Answer: {response}")
```

## Conclusion

Ollama Cloud support in LLM Client provides:
- Seamless integration with existing code
- Automatic detection from model names
- Easy switching between local and cloud
- Same API for both modes

Try it out with:
```bash
pip install llm_client
export OLLAMA_API_KEY=your_key
```

```python
from llm_client import LLMClient

client = LLMClient(llm="gpt-oss:120b-cloud")
response = client.chat_completion([
    {"role": "user", "content": "Hello, Ollama Cloud!"}
])
print(response)
```
