# Google Gemini Provider

The Gemini provider enables access to Google's Gemini models through the OpenAI compatibility API.

## Setup

### 1. Get API Key

1. Visit [Google AI Studio](https://aistudio.google.com/apikey)  
2. Sign in with your Google account  
3. Click **"Get API Key"** or **"Create API Key"**  
4. Select or create a Google Cloud project  
5. Copy the generated key (starts with `AIzaSy...`)  

### 2. Configure

```bash
# In secrets.env or environment
GEMINI_API_KEY=AIzaSy-your-api-key-here
```

## Usage

### Basic Usage

```python
from llm_client import LLMClient

# Explicit selection
client = LLMClient(api_choice="gemini")
```

### Available Models

Based on Google Gemini API documentation (December 2025):

**Stable Production Models:**

| Model | Description | Context Window | Best For |
|-------|-------------|----------------|----------|
| `gemini-2.5-pro` | Highest performance | 2M tokens | Complex reasoning, long documents |
| `gemini-2.5-flash` | Optimal balance | 1M tokens | General-purpose tasks |
| `gemini-2.5-flash-lite` | Massive scale | 1M tokens | High-throughput applications |
| `gemini-2.0-flash` | Cost-effective | 1M tokens | Budget-conscious deployments |

**Experimental/Preview Models:**

| Model | Description | Context Window | Notes |
|-------|-------------|----------------|-------|
| `gemini-3-pro` | Latest with extended reasoning | 2M tokens | Preview - may change |
| `gemini-3.1-flash-lite-preview` | Experimental Flash | 1M tokens | Testing new features |

### Model Selection

```python
# Use default model (gemini-3.1-flash-lite-preview)
client = LLMClient(api_choice="gemini")

# Specify model
client = LLMClient(
    api_choice="gemini",
    llm="gemini-2.5-pro"
)

# With custom parameters
client = LLMClient(
    api_choice="gemini",
    llm="gemini-2.5-flash",
    temperature=0.8,
    max_tokens=2048
)
```

## Features

### Chat Completion

```python
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain quantum entanglement."}
]

response = client.chat_completion(messages)
print(response)
```

### Streaming

```python
messages = [
    {"role": "user", "content": "Write a poem about artificial intelligence"}
]

print("Response: ", end="")
for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
print()
```

### Function Calling

Gemini supports OpenAI-compatible function calling:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search internal knowledge base",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "category": {
                    "type": "string",
                    "enum": ["technical", "business", "research"]
                }
            },
            "required": ["query"]
        }
    }
}]

messages = [
    {"role": "user", "content": "Find technical docs about RAG"}
]

result = client.chat_completion_with_tools(messages, tools)

if result['tool_calls']:
    for call in result['tool_calls']:
        print(f"Calling: {call['function']['name']}")
        print(f"Arguments: {call['function']['arguments']}")
```

### Long Context Processing

Gemini excels at processing very long documents:

```python
# Load large document
with open("long_document.txt", "r") as f:
    document = f.read()

# Gemini can handle up to 2M tokens
client = LLMClient(
    api_choice="gemini",
    llm="gemini-2.5-pro",
    max_tokens=4096
)

messages = [
    {"role": "system", "content": "You are a document analyzer."},
    {"role": "user", "content": f"Summarize this document:\n\n{document}"}
]

summary = client.chat_completion(messages)
print(summary)
```

## Configuration

### Via Config File

```yaml
# llm_config.yaml
providers:
  gemini:
    model: gemini-2.5-flash
    temperature: 0.8
    max_tokens: 2048
```

```python
client = LLMClient.from_config("llm_config.yaml", provider="gemini")
```

### Runtime Parameters

```python
client = LLMClient(
    api_choice="gemini",
    llm="gemini-2.5-pro",
    temperature=0.7,      # 0.0 = focused, 2.0 = creative
    max_tokens=2048       # Maximum response length
)
```

## Best Practices

### 1. Choose the Right Model

```python
# For complex reasoning - use gemini-2.5-pro
client = LLMClient(api_choice="gemini", llm="gemini-2.5-pro")
complex_response = client.chat_completion([
    {"role": "user", "content": "Analyze the geopolitical implications..."}
])

# For general tasks - use gemini-2.5-flash (faster, cheaper)
client.switch_provider("gemini", llm="gemini-2.5-flash")
quick_response = client.chat_completion([
    {"role": "user", "content": "Translate this text..."}
])

# For high throughput - use gemini-2.5-flash-lite
client.switch_provider("gemini", llm="gemini-2.5-flash-lite")
```

### 2. Leverage Long Context

```python
# Gemini handles very long contexts efficiently
client = LLMClient(
    api_choice="gemini",
    llm="gemini-2.5-pro"
)

# Count tokens before sending
from llm_client import TokenCounter

token_count = TokenCounter.count_tokens(messages)
print(f"Tokens: {token_count}")

# Gemini 2.5 Pro supports up to 2M tokens
if token_count < 2_000_000:
    response = client.chat_completion(messages)
```

### 3. Multimodal Capabilities

While not directly supported through the OpenAI compatibility API used by llm_client, Gemini natively supports image and video input through the Google AI SDK.

### 4. Streaming for Long Responses

```python
# Use streaming for better UX with long outputs
messages = [
    {"role": "user", "content": "Write a detailed analysis of..."}
]

for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
```

### 5. Temperature Control

```python
# Low temperature for factual responses
factual_client = LLMClient(
    api_choice="gemini",
    llm="gemini-2.5-flash",
    temperature=0.2
)

# High temperature for creative content
creative_client = LLMClient(
    api_choice="gemini",
    llm="gemini-2.5-flash",
    temperature=1.5
)
```

## Async Support

```python
import asyncio
from llm_client import LLMClient

async def main():
    client = LLMClient(
        api_choice="gemini",
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
    client = LLMClient(api_choice="gemini")
    response = client.chat_completion(messages)
except APIKeyNotFoundError:
    print("Gemini API key not found!")
    print("Set GEMINI_API_KEY environment variable")
except ChatCompletionError as e:
    print(f"API call failed: {e}")
    print(f"Original error: {e.original_error}")
```

## Pricing

Google Gemini offers competitive pricing with a generous free tier:

**Free Tier:**  
- 15 requests per minute  
- 1 million tokens per minute  
- 1,500 requests per day  

**Paid Tier (Pay-as-you-go):**

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Gemini 2.5 Pro | $1.25 | $5.00 |
| Gemini 2.5 Flash | $0.075 | $0.30 |
| Gemini 2.5 Flash Lite | $0.0375 | $0.15 |
| Gemini 2.0 Flash | $0.075 | $0.30 |

Check [Google AI Pricing](https://ai.google.dev/pricing) for current rates.

### Cost Estimation

```python
from llm_client import TokenCounter

messages = [
    {"role": "user", "content": "Analyze this data..."}
]

token_count = TokenCounter.count_tokens(messages)
estimated_response = 500

# For gemini-2.5-flash
input_cost = (token_count / 1_000_000) * 0.075
output_cost = (estimated_response / 1_000_000) * 0.30
total = input_cost + output_cost

print(f"Estimated cost: ${total:.4f}")
```

## Comparison with Other Providers

**Advantages:**  
- ✅ Very long context windows (up to 2M tokens)  
- ✅ Competitive pricing  
- ✅ Strong multilingual capabilities  
- ✅ Excellent at structured data extraction  
- ✅ Native multimodal support (via Google AI SDK)  

**Considerations:**  
- ⚠️ Newer than GPT-4, ecosystem still developing  
- ⚠️ Some features require Google AI SDK (not OpenAI compatibility)  
- ⚠️ Regional availability may vary  

## Troubleshooting

### API Key Issues

```bash
# Verify key is set
echo $GEMINI_API_KEY

# Or in Python
import os
print(os.getenv("GEMINI_API_KEY"))
```

### Rate Limit Errors

```python
import time
from llm_client.exceptions import ChatCompletionError

for attempt in range(3):
    try:
        response = client.chat_completion(messages)
        break
    except ChatCompletionError as e:
        if "rate_limit" in str(e).lower():
            wait_time = (attempt + 1) * 10
            print(f"Rate limit hit, waiting {wait_time}s...")
            time.sleep(wait_time)
        else:
            raise
```

### Context Length Errors

```python
from llm_client import TokenCounter

token_count = TokenCounter.count_tokens(messages)
model_limit = 1_000_000  # gemini-2.5-flash limit

if token_count > model_limit:
    print(f"Message too long: {token_count} > {model_limit}")
    # Consider using gemini-2.5-pro (2M limit)
    client.switch_provider("gemini", llm="gemini-2.5-pro")
```

## Resources

- [Google AI Studio](https://aistudio.google.com/)  
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)  
- [OpenAI Compatibility Guide](https://ai.google.dev/gemini-api/docs/openai)  
- [Pricing Information](https://ai.google.dev/pricing)  
- [Model Comparison](https://ai.google.dev/gemini-api/docs/models/gemini)  

## Example: Complete Workflow

```python
from llm_client import LLMClient
from llm_client.exceptions import ChatCompletionError

# Initialize client
client = LLMClient(
    api_choice="gemini",
    llm="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=1024
)

# Multi-turn conversation
conversation = [
    {"role": "system", "content": "You are a helpful research assistant."},
    {"role": "user", "content": "What are the latest trends in AI?"}
]

try:
    # Get initial response
    response = client.chat_completion(conversation)
    print(f"Assistant: {response}\n")

    # Continue conversation
    conversation.append({"role": "assistant", "content": response})
    conversation.append({"role": "user", "content": "Can you elaborate on transformers?"})

    # Stream the follow-up response
    print("Assistant: ", end="")
    for chunk in client.chat_completion_stream(conversation):
        print(chunk, end="", flush=True)
    print("\n")

except ChatCompletionError as e:
    print(f"Error: {e}")
```
