# OpenAI Provider

The OpenAI provider enables access to OpenAI's GPT models including GPT-4o, GPT-4o-mini, and GPT-3.5-turbo.

## Setup

### 1. Get API Key

1. Create account at [platform.openai.com](https://platform.openai.com)
2. Navigate to [API Keys](https://platform.openai.com/account/api-keys)
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)

### 2. Configure

```bash
# In secrets.env or environment
OPENAI_API_KEY=sk-your-api-key-here
```

## Usage

### Basic Usage

```python
from llm_client import LLMClient

# Auto-select (uses OpenAI if key is set)
client = LLMClient()

# Explicit selection
client = LLMClient(api_choice="openai")
```

### Available Models

| Model | Description | Context Window |
|-------|-------------|----------------|
| `gpt-4o` | Most capable model | 128K tokens |
| `gpt-4o-mini` | Fast, cost-effective (default) | 128K tokens |
| `gpt-3.5-turbo` | Legacy model | 16K tokens |

### Model Selection

```python
# Use default model (gpt-4o-mini)
client = LLMClient(api_choice="openai")

# Specify model
client = LLMClient(
    api_choice="openai",
    llm="gpt-4o"
)

# With parameters
client = LLMClient(
    api_choice="openai",
    llm="gpt-4o",
    temperature=0.7,
    max_tokens=2048
)
```

## Features

### Chat Completion

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain machine learning."}
]

response = client.chat_completion(messages)
print(response)
```

### Streaming

```python
messages = [
    {"role": "user", "content": "Write a poem about AI"}
]

print("Response: ", end="")
for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
print()
```

### Function Calling

OpenAI's function calling is fully supported:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
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

messages = [
    {"role": "user", "content": "What's the weather in Boston?"}
]

result = client.chat_completion_with_tools(messages, tools)

if result['tool_calls']:
    for call in result['tool_calls']:
        function_name = call['function']['name']
        arguments = call['function']['arguments']
        print(f"Calling: {function_name}({arguments})")
```

### Token Counting

```python
messages = [
    {"role": "user", "content": "Hello, how are you?"}
]

# Count tokens
token_count = client.count_tokens(messages)
print(f"Tokens: {token_count}")

# Check budget
max_tokens = 4096
reserved_for_response = 500

if token_count + reserved_for_response < max_tokens:
    response = client.chat_completion(messages)
```

## Configuration

### Via Config File

```yaml
# llm_config.yaml
default_provider: openai

providers:
  openai:
    model: gpt-4o-mini
    temperature: 0.7
    max_tokens: 512
```

```python
client = LLMClient.from_config("llm_config.yaml")
```

### Runtime Parameters

```python
client = LLMClient(
    api_choice="openai",
    llm="gpt-4o",
    temperature=0.5,      # 0.0 = deterministic, 2.0 = very random
    max_tokens=2048       # Maximum response length
)
```

## Async Support

```python
import asyncio

async def main():
    client = LLMClient(
        api_choice="openai",
        use_async=True
    )

    messages = [{"role": "user", "content": "Hello"}]

    # Async completion
    response = await client.achat_completion(messages)
    print(response)

    # Async streaming
    async for chunk in client.achat_completion_stream(messages):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

## Error Handling

```python
from llm_client.exceptions import (
    APIKeyNotFoundError,
    ChatCompletionError
)

try:
    client = LLMClient(api_choice="openai")
    response = client.chat_completion(messages)
except APIKeyNotFoundError:
    print("OpenAI API key not found!")
    print("Set OPENAI_API_KEY environment variable")
except ChatCompletionError as e:
    print(f"API call failed: {e}")
    print(f"Original error: {e.original_error}")
```

## Best Practices

### 1. Choose the Right Model

```python
# For simple tasks - use gpt-4o-mini (faster, cheaper)
client = LLMClient(api_choice="openai", llm="gpt-4o-mini")
simple_response = client.chat_completion([
    {"role": "user", "content": "What is 2+2?"}
])

# For complex tasks - use gpt-4o (more capable)
client.switch_provider("openai", llm="gpt-4o")
complex_response = client.chat_completion([
    {"role": "user", "content": "Analyze this complex data..."}
])
```

### 2. Manage Token Usage

```python
# Count tokens before API call
token_count = client.count_tokens(messages)

if token_count > 3000:
    print("Warning: Large input, may be slow/expensive")

# Leave room for response
max_input = 4096 - 500  # Reserve 500 tokens for response
if token_count < max_input:
    response = client.chat_completion(messages)
```

### 3. Handle Rate Limits

The client automatically retries with exponential backoff:

```python
# Automatic retry on transient failures
response = client.chat_completion(messages)
# Up to 3 retries with delays: 4s, 8s, 10s
```

### 4. Use Streaming for Long Responses

```python
# Streaming provides better UX for long responses
for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
```

### 5. System Messages

```python
messages = [
    {
        "role": "system",
        "content": "You are a Python expert. Provide concise answers."
    },
    {
        "role": "user",
        "content": "How do I read a file in Python?"
    }
]
```

## Troubleshooting

### API Key Issues

```bash
# Verify key is set
echo $OPENAI_API_KEY

# Or in Python
import os
print(os.getenv("OPENAI_API_KEY"))
```

### Rate Limit Errors

If you hit rate limits, the client automatically retries. For persistent issues:

```python
import time

for attempt in range(3):
    try:
        response = client.chat_completion(messages)
        break
    except ChatCompletionError as e:
        if "rate_limit" in str(e).lower():
            time.sleep(10 * (attempt + 1))  # Increasing backoff
        else:
            raise
```

### Context Length Errors

```python
# Check if message fits in context
token_count = client.count_tokens(messages)
model_limit = 128000  # gpt-4o limit

if token_count > model_limit:
    print(f"Message too long: {token_count} > {model_limit}")
    # Truncate or summarize messages
```

## Pricing

Approximate pricing (check [OpenAI pricing page](https://openai.com/pricing) for current rates):

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| GPT-4o | $2.50 | $10.00 |
| GPT-4o-mini | $0.15 | $0.60 |
| GPT-3.5-turbo | $0.50 | $1.50 |

### Cost Estimation

```python
# Estimate cost
token_count = client.count_tokens(messages)
estimated_response_tokens = 200

# For gpt-4o-mini
input_cost = (token_count / 1_000_000) * 0.15
output_cost = (estimated_response_tokens / 1_000_000) * 0.60
total_cost = input_cost + output_cost

print(f"Estimated cost: ${total_cost:.4f}")
```

## Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [OpenAI Playground](https://platform.openai.com/playground)
- [Model Pricing](https://openai.com/pricing)
- [API Status](https://status.openai.com/)
