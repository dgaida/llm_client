# Token Counting

Accurate token counting helps you manage API costs and stay within context limits. LLM Client uses [tiktoken](https://github.com/openai/tiktoken) for precise token counting across all GPT models.

---

## Overview

**Why Token Counting Matters:**

- 💰 **Cost Management** - Know exactly how much each request will cost
- 📊 **Context Limits** - Ensure messages fit within model limits
- 🎯 **Optimization** - Identify and reduce unnecessary tokens
- 📈 **Monitoring** - Track token usage over time

---

## Basic Usage

### Count Tokens in Messages

```python
from llm_client import LLMClient

client = LLMClient()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is quantum computing?"}
]

# Count tokens
token_count = client.count_tokens(messages)
print(f"Total tokens: {token_count}")
```

### Count Tokens in a String

```python
text = "Hello, how are you doing today?"
tokens = client.count_string_tokens(text)
print(f"String has {tokens} tokens")
```

---

## Budget Management

### Check Before Sending

```python
from llm_client import LLMClient

client = LLMClient()
messages = [{"role": "user", "content": "Explain AI in detail"}]

# Count input tokens
input_tokens = client.count_tokens(messages)
print(f"Input: {input_tokens} tokens")

# Reserve tokens for response
max_context = 4096
reserved_for_response = 500
available = max_context - input_tokens - reserved_for_response

if available > 0:
    print(f"✓ {available} tokens available for response")
    response = client.chat_completion(messages)
else:
    print("✗ Message too long! Reduce input.")
```

### Estimate Total Cost

```python
from llm_client import LLMClient

client = LLMClient(api_choice="openai", llm="gpt-4o-mini")

messages = [
    {"role": "user", "content": "Write a detailed essay about climate change"}
]

# Count input tokens
input_tokens = client.count_tokens(messages)

# Estimate response tokens (adjust based on your use case)
estimated_response_tokens = 500

# Calculate cost (gpt-4o-mini pricing)
input_cost_per_1m = 0.15  # $0.15 per 1M input tokens
output_cost_per_1m = 0.60  # $0.60 per 1M output tokens

input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
output_cost = (estimated_response_tokens / 1_000_000) * output_cost_per_1m
total_cost = input_cost + output_cost

print(f"Estimated cost: ${total_cost:.6f}")
```

---

## Advanced Usage

### Per-Model Token Counting

```python
# Count tokens for different models
messages = [{"role": "user", "content": "Hello world"}]

gpt4o_tokens = client.count_tokens(messages, model="gpt-4o")
gpt4o_mini_tokens = client.count_tokens(messages, model="gpt-4o-mini")
gpt35_tokens = client.count_tokens(messages, model="gpt-3.5-turbo")

print(f"GPT-4o: {gpt4o_tokens} tokens")
print(f"GPT-4o-mini: {gpt4o_mini_tokens} tokens")
print(f"GPT-3.5: {gpt35_tokens} tokens")
```

### Token Counter Class

For standalone token counting without a client:

```python
from llm_client import TokenCounter

counter = TokenCounter()

# Count tokens in messages
messages = [{"role": "user", "content": "Hello"}]
count = counter.count_tokens(messages, model="gpt-4o")

# Count tokens in string
text = "Hello world"
count = counter.count_string_tokens(text, model="gpt-4o")

# Check if tiktoken is available
if counter.is_tiktoken_available():
    print("Using accurate tiktoken counting")
else:
    print("Using fallback estimation")
```

---

## Conversation Management

### Track Conversation Length

```python
from llm_client import LLMClient

client = LLMClient()
conversation = [
    {"role": "system", "content": "You are a helpful assistant."}
]

max_tokens = 4096
reserved = 500

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # Add user message
    conversation.append({"role": "user", "content": user_input})

    # Check token count
    current_tokens = client.count_tokens(conversation)
    print(f"Conversation: {current_tokens} tokens")

    if current_tokens + reserved > max_tokens:
        # Remove oldest messages to stay within limit
        print("⚠️ Conversation too long, removing old messages")
        conversation = [conversation[0]] + conversation[-5:]

    # Get response
    response = client.chat_completion(conversation)
    conversation.append({"role": "assistant", "content": response})

    print(f"Assistant: {response}")
```

### Sliding Window

```python
def maintain_conversation_window(
    conversation: list[dict],
    max_tokens: int = 4000,
    keep_system: bool = True
) -> list[dict]:
    """Keep conversation within token limit using sliding window."""

    counter = TokenCounter()

    # Always keep system message if present
    if keep_system and conversation and conversation[0]["role"] == "system":
        system_msg = [conversation[0]]
        messages = conversation[1:]
    else:
        system_msg = []
        messages = conversation

    # Count tokens and remove oldest messages if needed
    while counter.count_tokens(system_msg + messages) > max_tokens:
        if len(messages) <= 2:  # Keep at least last 2 messages
            break
        messages = messages[1:]  # Remove oldest message

    return system_msg + messages

# Usage
conversation = maintain_conversation_window(conversation, max_tokens=4000)
```

---

## Model Encodings

LLM Client supports accurate token counting for all GPT models:

| Model | Encoding | Supported |
|-------|----------|-----------|
| `gpt-4o` | `o200k_base` | ✅ |
| `gpt-4o-mini` | `o200k_base` | ✅ |
| `gpt-4` | `cl100k_base` | ✅ |
| `gpt-3.5-turbo` | `cl100k_base` | ✅ |

For non-GPT models (Groq, Gemini, Ollama), token counting uses GPT-4o encoding as an approximation.

---

## Fallback Estimation

If tiktoken is not installed, LLM Client falls back to rough estimation:

```python
# Estimation when tiktoken not available
# Approximation: 1 token ≈ 4 characters
```

**Install tiktoken for accurate counting:**

```bash
pip install tiktoken
```

---

## Best Practices

### 1. Count Before Every Request

```python
# ✅ Good: Check token count
token_count = client.count_tokens(messages)
if token_count < 4000:
    response = client.chat_completion(messages)

# ❌ Bad: Assume it fits
response = client.chat_completion(messages)
```

### 2. Reserve Tokens for Response

```python
# ✅ Good: Reserve space for response
max_input = 4096 - 500  # Reserve 500 tokens
if client.count_tokens(messages) < max_input:
    response = client.chat_completion(messages)

# ❌ Bad: Use full context window
if client.count_tokens(messages) < 4096:
    response = client.chat_completion(messages)
```

### 3. Monitor Token Usage

```python
import logging

def chat_with_monitoring(client, messages):
    """Chat with token usage monitoring."""

    input_tokens = client.count_tokens(messages)
    logging.info(f"Input tokens: {input_tokens}")

    response = client.chat_completion(messages)

    output_tokens = client.count_string_tokens(response)
    logging.info(f"Output tokens: {output_tokens}")
    logging.info(f"Total tokens: {input_tokens + output_tokens}")

    return response
```

### 4. Optimize Long Conversations

```python
# ✅ Good: Summarize old messages
if token_count > 3000:
    # Summarize conversation history
    summary = client.chat_completion([
        {"role": "user", "content": f"Summarize: {old_messages}"}
    ])
    conversation = [
        {"role": "system", "content": summary},
        *recent_messages
    ]

# ❌ Bad: Keep all messages
conversation.append(new_message)
```

---

## API Reference

### LLMClient.count_tokens()

```python
def count_tokens(
    messages: list[dict[str, str]],
    model: str | None = None
) -> int:
    """Count tokens in messages.

    Args:
        messages: List of message dicts
        model: Model name. If None, uses current model.

    Returns:
        Total token count
    """
```

### LLMClient.count_string_tokens()

```python
def count_string_tokens(
    text: str,
    model: str | None = None
) -> int:
    """Count tokens in a string.

    Args:
        text: Text to count
        model: Model name. If None, uses current model.

    Returns:
        Token count
    """
```

### TokenCounter

```python
from llm_client import TokenCounter

counter = TokenCounter()

# Count tokens in messages
count = counter.count_tokens(messages, model="gpt-4o")

# Count tokens in string
count = counter.count_string_tokens(text, model="gpt-4o")

# Check availability
available = counter.is_tiktoken_available()
```

---

## Related

- [API Reference](../api_reference.md) - Complete API documentation
- [Cost Optimization](../examples/provider_switching.md#cost-optimization) - Strategies for reducing costs
- [Troubleshooting](../troubleshooting.md#token-counting) - Common token counting issues
