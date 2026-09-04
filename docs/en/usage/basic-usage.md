# Basic Usage Examples

## Simple Chat Completion

```python
from llm_client import LLMClient

# Create client (auto-selects API)
client = LLMClient()

# Single message
messages = [
    {"role": "user", "content": "What is Python?"}
]

response = client.chat_completion(messages)
print(response)
```

## Multi-Turn Conversation

```python
from llm_client import LLMClient

client = LLMClient()

# Start conversation
conversation = [
    {"role": "system", "content": "You are a helpful programming assistant."},
    {"role": "user", "content": "What is a list in Python?"}
]

# Get first response
response1 = client.chat_completion(conversation)
print(f"Assistant: {response1}\n")

# Add to conversation history
conversation.append({"role": "assistant", "content": response1})

# Continue conversation
conversation.append({"role": "user", "content": "Can you give me an example?"})
response2 = client.chat_completion(conversation)
print(f"Assistant: {response2}")
```

## Streaming Responses

```python
from llm_client import LLMClient

client = LLMClient()

messages = [
    {"role": "user", "content": "Write a short story about a robot"}
]

print("Assistant: ", end="")
for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
print("\n")
```

## Listing Available Models

```python
from llm_client import LLMClient

client = LLMClient(api_choice="openai")

# List models available for the active provider
models = client.list_models()
print("Available models:", models)
```

## Using Different Models

```python
from llm_client import LLMClient

# GPT-4o for complex tasks
gpt4_client = LLMClient(
    api_choice="openai",
    llm="gpt-4o"
)

complex_response = gpt4_client.chat_completion([
    {"role": "user", "content": "Explain quantum entanglement"}
])

# GPT-4o-mini for simple tasks
mini_client = LLMClient(
    api_choice="openai",
    llm="gpt-4o-mini"
)

simple_response = mini_client.chat_completion([
    {"role": "user", "content": "What is 2+2?"}
])
```

## Adjusting Temperature

```python
from llm_client import LLMClient

# Low temperature (deterministic, focused)
deterministic = LLMClient(temperature=0.1)
response1 = deterministic.chat_completion([
    {"role": "user", "content": "List the planets"}
])

# High temperature (creative, varied)
creative = LLMClient(temperature=1.5)
response2 = creative.chat_completion([
    {"role": "user", "content": "Write a creative story opening"}
])
```

## Controlling Response Length

```python
from llm_client import LLMClient

# Short response
short_client = LLMClient(max_tokens=50)
short_response = short_client.chat_completion([
    {"role": "user", "content": "Explain AI in one sentence"}
])

# Long response
long_client = LLMClient(max_tokens=500)
long_response = long_client.chat_completion([
    {"role": "user", "content": "Explain AI in detail"}
])
```

## System Messages

```python
from llm_client import LLMClient

client = LLMClient()

# Technical expert persona
messages = [
    {
        "role": "system",
        "content": "You are a senior software engineer with expertise in Python."
    },
    {
        "role": "user",
        "content": "How do I optimize this code?"
    }
]

response = client.chat_completion(messages)
```

## Error Handling

```python
from llm_client import LLMClient
from llm_client.exceptions import (
    APIKeyNotFoundError,
    ChatCompletionError,
    InvalidProviderError
)

try:
    client = LLMClient(api_choice="openai")
    response = client.chat_completion(messages)
    print(response)

except APIKeyNotFoundError as e:
    print(f"Missing API key: {e.key_name}")
    print("Please set your API key in environment")

except ChatCompletionError as e:
    print(f"API call failed: {e}")
    print(f"Provider: {e.provider}")
    print(f"Original error: {e.original_error}")

except InvalidProviderError as e:
    print(f"Invalid provider: {e.provider}")
    print(f"Valid providers: {e.valid_providers}")
```

## Token Counting

```python
from llm_client import LLMClient

client = LLMClient()

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Explain machine learning"}
]

# Count tokens before sending
token_count = client.count_tokens(messages)
print(f"Input tokens: {token_count}")

# Check if within budget
max_context = 4096
reserved_for_response = 500

if token_count + reserved_for_response < max_context:
    response = client.chat_completion(messages)

    # Estimate response tokens
    response_tokens = client.count_string_tokens(response)
    print(f"Response tokens: {response_tokens}")
    print(f"Total tokens: {token_count + response_tokens}")
else:
    print("Message too long!")
```

## Using Configuration Files

```python
from llm_client import LLMClient

# Load from config file
client = LLMClient.from_config("llm_config.yaml")

# Config file automatically sets model, temperature, etc.
response = client.chat_completion(messages)
```

Example `llm_config.yaml`:

```yaml
default_provider: openai

providers:
  openai:
    model: gpt-4o-mini
    temperature: 0.7
    max_tokens: 512
```

## Async Operations

```python
import asyncio
from llm_client import LLMClient

async def main():
    # Create async client
    client = LLMClient(use_async=True)

    messages = [
        {"role": "user", "content": "What is async programming?"}
    ]

    # Async completion
    response = await client.achat_completion(messages)
    print(response)

    # Async streaming
    print("\nStreaming:")
    async for chunk in client.achat_completion_stream(messages):
        print(chunk, end="", flush=True)
    print()

# Run async code
asyncio.run(main())
```

## Processing Multiple Queries

```python
from llm_client import LLMClient

client = LLMClient()

questions = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?"
]

for question in questions:
    print(f"\nQ: {question}")
    response = client.chat_completion([
        {"role": "user", "content": question}
    ])
    print(f"A: {response[:100]}...")  # First 100 chars
```

## Concurrent Async Requests

```python
import asyncio
from llm_client import LLMClient

async def process_queries():
    client = LLMClient(use_async=True)

    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?"
    ]

    # Create concurrent tasks
    tasks = [
        client.achat_completion([{"role": "user", "content": q}])
        for q in questions
    ]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Print results
    for q, r in zip(questions, responses):
        print(f"\nQ: {q}")
        print(f"A: {r[:100]}...")

asyncio.run(process_queries())
```
