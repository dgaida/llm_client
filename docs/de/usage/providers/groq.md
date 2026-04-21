# Groq Provider

The Groq provider offers ultra-fast inference on open-source models through GroqCloud.

## Setup

### 1. Get API Key

1. Create account at [console.groq.com](https://console.groq.com)  
2. Navigate to [API Keys](https://console.groq.com/keys)  
3. Click "Create API Key"  
4. Copy the key (starts with `gsk_...`)  

### 2. Configure

```bash
# In secrets.env or environment
GROQ_API_KEY=gsk-your-api-key-here
```

## Usage

### Basic Usage

```python
from llm_client import LLMClient

# Explicit selection
client = LLMClient(api_choice="groq")
```

### Available Models

| Model | Description | Context Window | Speed |
|-------|-------------|----------------|-------|
| `llama-3.3-70b-versatile` | Llama 3.3 70B | 128K | Very Fast |
| `meta-llama/llama-4-maverick-17b-128e-instruct` | Llama 4 17B | 128K | Ultra Fast |
| `openai/gpt-oss-120b` | GPT OSS 120B | 128K | Fast |
| `qwen/qwen3-32b` | Qwen 3 (default) | 128K | Fast |

### Model Selection

```python
# Use default model
client = LLMClient(api_choice="groq")

# Specify model
client = LLMClient(
    api_choice="groq",
    llm="llama-3.3-70b-versatile"
)
```

## Features

### Ultra-Fast Inference

```python
import time

messages = [{"role": "user", "content": "Explain quantum computing"}]

start = time.time()
response = client.chat_completion(messages)
elapsed = time.time() - start

print(f"Response in {elapsed:.2f}s")
# Typical: 0.3-1.0 seconds
```

### Streaming

```python
messages = [{"role": "user", "content": "Count from 1 to 10"}]

for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
print()
```

### Function Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            }
        }
    }
}]

result = client.chat_completion_with_tools(messages, tools)
```

## Configuration

### Via Config File

```yaml
# llm_config.yaml
providers:
  groq:
    model: llama-3.3-70b-versatile
    temperature: 0.5
    max_tokens: 1024
```

### Runtime Parameters

```python
client = LLMClient(
    api_choice="groq",
    llm="llama-3.3-70b-versatile",
    temperature=0.5,
    max_tokens=1024
)
```

## Best Practices

### 1. Leverage Speed

```python
# Groq is ideal for high-throughput applications
client = LLMClient(api_choice="groq")

queries = ["Question 1", "Question 2", "Question 3"]

for query in queries:
    response = client.chat_completion([
        {"role": "user", "content": query}
    ])
    print(f"Q: {query}\nA: {response}\n")
```

### 2. Cost-Effective for Volume

```python
# Use Groq for cost-sensitive applications
client = LLMClient(
    api_choice="groq",
    llm="llama-3.3-70b-versatile"
)

# Process large batches efficiently
for item in large_dataset:
    response = client.chat_completion(
        [{"role": "user", "content": item}]
    )
```

### 3. Fallback Strategy

```python
from llm_client.exceptions import ChatCompletionError

# Try Groq first (fast, cheap), fallback to OpenAI if needed
client = LLMClient(api_choice="groq")

try:
    response = client.chat_completion(messages)
except ChatCompletionError:
    # Fallback to OpenAI
    client.switch_provider("openai")
    response = client.chat_completion(messages)
```

## Async Support

```python
import asyncio

async def process_queries():
    client = LLMClient(api_choice="groq", use_async=True)

    queries = ["Q1", "Q2", "Q3"]

    tasks = [
        client.achat_completion([{"role": "user", "content": q}])
        for q in queries
    ]

    responses = await asyncio.gather(*tasks)
    return responses

# Process multiple queries concurrently
results = asyncio.run(process_queries())
```

## Pricing

Groq offers very competitive pricing for open-source models. Check [groq.com/pricing](https://groq.com/pricing) for current rates.

**Free Tier:**  
- Generous free tier for testing and development  
- Rate limits apply  

## Resources

- [Groq Documentation](https://console.groq.com/docs)  
- [Available Models](https://console.groq.com/docs/models)  
- [API Reference](https://console.groq.com/docs/api-reference)  
- [Pricing](https://groq.com/pricing)  
