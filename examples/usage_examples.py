"""Demonstration of new LLM Client features.

This script showcases:
1. Token counting with tiktoken
2. Async support for chat completions
3. Configuration file loading
"""

import asyncio
from pathlib import Path

from llm_client import LLMClient
from llm_client.config import generate_config_template


def example_token_counting():
    """Example 1: Token counting."""
    print("=" * 60)
    print("Example 1: Token Counting")
    print("=" * 60)

    client = LLMClient(api_choice="openai")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
    ]

    # Count tokens before sending
    token_count = client.count_tokens(messages)
    print(f"Message tokens: {token_count}")

    # Count tokens in a single string
    text = "This is a test message for token counting."
    string_tokens = client.count_string_tokens(text)
    print(f"String '{text}' has {string_tokens} tokens")

    # Check if tiktoken is available
    from llm_client.token_counter import TokenCounter

    if TokenCounter.is_tiktoken_available():
        print("✓ Using accurate tiktoken counting")
    else:
        print("⚠ Using rough token estimation (install tiktoken for accuracy)")

    print()


async def example_async_chat():
    """Example 2: Async chat completion."""
    print("=" * 60)
    print("Example 2: Async Chat Completion")
    print("=" * 60)

    # Create async client
    client = LLMClient(api_choice="openai", use_async=True)

    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Explain async programming in one sentence."},
    ]

    print("Sending async request...")
    response = await client.achat_completion(messages)
    print(f"Response: {response}")
    print()


async def example_async_streaming():
    """Example 3: Async streaming."""
    print("=" * 60)
    print("Example 3: Async Streaming")
    print("=" * 60)

    client = LLMClient(api_choice="openai", use_async=True)

    messages = [{"role": "user", "content": "Count from 1 to 5, one number per line."}]

    print("Streaming async response:")
    async for chunk in client.achat_completion_stream(messages):
        print(chunk, end="", flush=True)
    print("\n")


def example_config_file():
    """Example 4: Configuration file loading."""
    print("=" * 60)
    print("Example 4: Configuration File Loading")
    print("=" * 60)

    # Generate template config file
    config_path = Path("llm_config_demo.yaml")
    if not config_path.exists():
        print("Generating config template...")
        generate_config_template(config_path, format="yaml")
        print(f"✓ Created {config_path}")

    # Load client from config (default provider)
    print("\nLoading client from config (default provider)...")
    client = LLMClient.from_config(config_path)
    print(f"Client: {client}")
    print(f"Provider: {client.api_choice}")
    print(f"Model: {client.llm}")
    print(f"Temperature: {client.temperature}")

    # Load specific provider from config
    print("\nLoading specific provider (groq)...")
    client_groq = LLMClient.from_config(config_path, provider="groq")
    print(f"Client: {client_groq}")
    print(f"Model: {client_groq.llm}")

    # Clean up
    if config_path.exists():
        config_path.unlink()
        print(f"\n✓ Cleaned up {config_path}")

    print()


def example_config_programmatic():
    """Example 5: Programmatic config creation."""
    print("=" * 60)
    print("Example 5: Programmatic Configuration")
    print("=" * 60)

    from llm_client.config import LLMConfig

    # Create config from dictionary
    config_dict = {
        "default_provider": "groq",
        "providers": {
            "groq": {"model": "llama-3.3-70b-versatile", "temperature": 0.3},
            "openai": {"model": "gpt-4o", "temperature": 0.7},
        },
    }

    config = LLMConfig.from_dict(config_dict)
    print(f"Config: {config}")
    print(f"Default provider: {config.default_provider}")
    print(f"Available providers: {config.list_providers()}")

    # Validate configuration
    is_valid, errors = config.validate()
    if is_valid:
        print("✓ Configuration is valid")
    else:
        print(f"✗ Configuration errors: {errors}")

    print()


def example_token_counting_with_models():
    """Example 6: Token counting for different models."""
    print("=" * 60)
    print("Example 6: Token Counting for Different Models")
    print("=" * 60)

    from llm_client.token_counter import TokenCounter

    text = "Hello, how are you doing today?"

    models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

    for model in models:
        token_count = TokenCounter.count_string_tokens(text, model=model)
        print(f"{model:20s}: {token_count} tokens")

    print()


async def example_concurrent_async_requests():
    """Example 7: Concurrent async requests."""
    print("=" * 60)
    print("Example 7: Concurrent Async Requests")
    print("=" * 60)

    client = LLMClient(api_choice="openai", use_async=True)

    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    print("Sending 3 concurrent requests...")

    # Create tasks for concurrent execution
    tasks = [client.achat_completion([{"role": "user", "content": q}]) for q in questions]

    # Wait for all to complete
    import time

    start = time.time()
    responses = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    print(f"\n✓ Completed in {elapsed:.2f}s")
    for q, r in zip(questions, responses, strict=False):
        print(f"\nQ: {q}")
        print(f"A: {r[:100]}...")

    print()


def example_token_budget_check():
    """Example 8: Check if message fits in token budget."""
    print("=" * 60)
    print("Example 8: Token Budget Checking")
    print("=" * 60)

    client = LLMClient()

    # Define token budget (e.g., model context limit)
    max_tokens = 4096
    reserved_for_response = 500

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in detail."},
    ]

    input_tokens = client.count_tokens(messages)
    available_tokens = max_tokens - input_tokens - reserved_for_response

    print(f"Input tokens: {input_tokens}")
    print(f"Reserved for response: {reserved_for_response}")
    print(f"Available for response: {available_tokens}")

    if available_tokens > 0:
        print("✓ Message fits in token budget")
    else:
        print("✗ Message exceeds token budget!")

    print()


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LLM Client - New Features Demonstration")
    print("=" * 60 + "\n")

    # Synchronous examples
    example_token_counting()
    example_token_counting_with_models()
    example_token_budget_check()
    example_config_file()
    example_config_programmatic()

    # Async examples
    print("Running async examples...\n")
    await example_async_chat()
    await example_async_streaming()
    await example_concurrent_async_requests()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
