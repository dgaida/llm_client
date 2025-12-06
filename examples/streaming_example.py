"""Example demonstrating streaming, retry logic, and error handling.

This script showcases the new features added to llm_client:
1. Response streaming
2. Automatic retry with exponential backoff
3. Custom exception handling
"""

import time

from llm_client import LLMClient
from llm_client.exceptions import (
    APIKeyNotFoundError,
    ChatCompletionError,
    InvalidProviderError,
    LLMClientError,
    StreamingNotSupportedError,
)


def example_basic_usage():
    """Example 1: Basic usage with automatic retry."""
    print("=" * 60)
    print("Example 1: Basic Chat Completion with Retry Logic")
    print("=" * 60)

    try:
        client = LLMClient(api_choice="openai")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning in one sentence?"},
        ]

        print("Sending request (with automatic retry on failures)...")
        response = client.chat_completion(messages)
        print(f"Response: {response}\n")

    except APIKeyNotFoundError as e:
        print(f"API Key Error: {e}\n")
    except ChatCompletionError as e:
        print(f"Chat Completion Error: {e}")
        print(f"Original error: {e.original_error}\n")
    except LLMClientError as e:
        print(f"General LLM Client Error: {e}\n")


def example_streaming():
    """Example 2: Streaming responses in real-time."""
    print("=" * 60)
    print("Example 2: Streaming Response")
    print("=" * 60)

    try:
        client = LLMClient(api_choice="openai")
        messages = [
            {"role": "system", "content": "You are a creative storyteller."},
            {"role": "user", "content": "Tell me a very short story about a robot."},
        ]

        print("Streaming response:\n")
        for chunk in client.chat_completion_stream(messages):
            print(chunk, end="", flush=True)
        print("\n")

    except StreamingNotSupportedError as e:
        print(f"Streaming Error: {e}")
        print("Falling back to regular completion...")
        response = client.chat_completion(messages)
        print(f"Response: {response}\n")

    except APIKeyNotFoundError as e:
        print(f"API Key Error: {e}\n")

    except ChatCompletionError as e:
        print(f"Chat Completion Error: {e}\n")


def example_provider_switching_with_streaming():
    """Example 3: Switching providers and using streaming."""
    print("=" * 60)
    print("Example 3: Provider Switching with Streaming")
    print("=" * 60)

    try:
        # Start with OpenAI
        client = LLMClient(api_choice="openai", llm="gpt-4o-mini")
        print(f"Current provider: {client.api_choice}")

        messages = [{"role": "user", "content": "Count from 1 to 5."}]

        print("Streaming from OpenAI:")
        for chunk in client.chat_completion_stream(messages):
            print(chunk, end="", flush=True)
        print("\n")

        # Switch to Groq
        print("Switching to Groq...")
        client.switch_provider("groq")
        print(f"Current provider: {client.api_choice}")

        print("Streaming from Groq:")
        for chunk in client.chat_completion_stream(messages):
            print(chunk, end="", flush=True)
        print("\n")

    except APIKeyNotFoundError as e:
        print(f"API Key Error: {e}\n")
    except InvalidProviderError as e:
        print(f"Invalid Provider: {e}\n")
    except LLMClientError as e:
        print(f"Error: {e}\n")


def example_error_handling():
    """Example 4: Comprehensive error handling."""
    print("=" * 60)
    print("Example 4: Error Handling")
    print("=" * 60)

    # Test invalid provider
    print("Test 1: Invalid provider name")
    try:
        client = LLMClient(api_choice="nonexistent")
    except InvalidProviderError as e:
        print(f"✓ Caught error: {e}")
        print(f"  Valid providers: {e.valid_providers}\n")

    # Test missing API key
    print("Test 2: Missing API key")
    try:
        import os

        # Temporarily remove API key
        original_key = os.environ.get("OPENAI_API_KEY")
        if original_key:
            del os.environ["OPENAI_API_KEY"]

        client = LLMClient(api_choice="openai")

        # Restore key
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key

    except APIKeyNotFoundError as e:
        print(f"✓ Caught error: {e}")
        print(f"  Provider: {e.provider}")
        print(f"  Required key: {e.key_name}\n")

        # Restore key
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key

    # Test switching to invalid provider
    print("Test 3: Switch to invalid provider")
    try:
        client = LLMClient()  # Use default provider
        client.switch_provider("invalid_api")
    except InvalidProviderError as e:
        print(f"✓ Caught error: {e}\n")


def example_retry_demonstration():
    """Example 5: Demonstrate retry logic (conceptual)."""
    print("=" * 60)
    print("Example 5: Retry Logic Demonstration")
    print("=" * 60)

    print("The retry logic is built into chat_completion():")
    print("- Automatically retries up to 3 times on failures")
    print("- Uses exponential backoff (4s, 8s, 10s)")
    print("- Transparently handles transient API errors")
    print()

    try:
        client = LLMClient()
        messages = [{"role": "user", "content": "Hello!"}]

        start_time = time.time()
        response = client.chat_completion(messages)
        elapsed = time.time() - start_time

        print(f"Response received in {elapsed:.2f}s")
        print(f"Response: {response[:100]}...")
        print("\nNote: If there were network issues, retries happened automatically!\n")

    except ChatCompletionError as e:
        print(f"Failed after retries: {e}\n")


def example_streaming_comparison():
    """Example 6: Compare streaming vs non-streaming."""
    print("=" * 60)
    print("Example 6: Streaming vs Non-Streaming Comparison")
    print("=" * 60)

    try:
        client = LLMClient()
        messages = [{"role": "user", "content": "List 3 benefits of streaming responses."}]

        # Non-streaming (waits for complete response)
        print("Non-streaming (complete response at once):")
        start = time.time()
        response = client.chat_completion(messages)
        elapsed = time.time() - start
        print(response)
        print(f"Time: {elapsed:.2f}s\n")

        # Streaming (tokens arrive progressively)
        print("Streaming (tokens arrive progressively):")
        start = time.time()
        for chunk in client.chat_completion_stream(messages):
            print(chunk, end="", flush=True)
        elapsed = time.time() - start
        print(f"\nTime: {elapsed:.2f}s")
        print("\nStreaming provides better UX for long responses!\n")

    except LLMClientError as e:
        print(f"Error: {e}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LLM Client - New Features Demonstration")
    print("=" * 60 + "\n")

    # Run examples
    example_basic_usage()
    example_streaming()
    example_provider_switching_with_streaming()
    example_error_handling()
    example_retry_demonstration()
    example_streaming_comparison()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
