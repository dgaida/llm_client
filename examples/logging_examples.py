"""Examples demonstrating logging functionality in llm_client.

This script shows various ways to configure and use logging with the LLMClient.
"""

import os
import sys

# Add parent directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_client import LLMClient, disable_logging, setup_logging


def example_basic_logging():
    """Basic logging example."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Logging (INFO level)")
    print("=" * 60)

    # Enable INFO level logging
    setup_logging(level="INFO")

    # Create client - will log provider selection
    client = LLMClient()

    # Make a simple request
    messages = [{"role": "user", "content": "Say hello in one word"}]
    response = client.chat_completion(messages)

    print(f"\nResponse: {response}")


def example_debug_logging():
    """Debug logging example - shows everything."""
    print("\n" + "=" * 60)
    print("Example 2: Debug Logging (DEBUG level)")
    print("=" * 60)

    # Enable DEBUG logging for maximum verbosity
    setup_logging(level="DEBUG", force=True)

    # Create client with specific provider
    client = LLMClient(api_choice="openai", llm="gpt-4o-mini")

    # Count tokens
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Explain AI briefly."},
    ]

    token_count = client.count_tokens(messages)
    print(f"\nToken count: {token_count}")

    # Make request
    response = client.chat_completion(messages)
    print(f"\nResponse: {response[:100]}...")


def example_provider_switching():
    """Example showing provider switching with logging."""
    print("\n" + "=" * 60)
    print("Example 3: Provider Switching with Logging")
    print("=" * 60)

    # Set to INFO level
    setup_logging(level="INFO", force=True)

    # Start with one provider
    client = LLMClient(api_choice="openai")
    print(f"Current provider: {client.api_choice}")

    # Switch to another provider
    if client.groq_api_key:
        client.switch_provider("groq", temperature=0.5)
        print(f"Switched to: {client.api_choice}")


def example_custom_format():
    """Example with custom log format."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Log Format")
    print("=" * 60)

    # Set up logging with custom format
    setup_logging(level="INFO", format_string="[%(levelname)s] %(name)s: %(message)s", force=True)

    client = LLMClient()
    messages = [{"role": "user", "content": "Hello"}]

    try:
        response = client.chat_completion(messages)
        print(f"\nResponse: {response}")
    except Exception as e:
        print(f"\nError: {e}")


def example_environment_variable():
    """Example using environment variable for log level."""
    print("\n" + "=" * 60)
    print("Example 5: Environment Variable Configuration")
    print("=" * 60)

    # Set environment variable
    os.environ["LLM_CLIENT_LOG_LEVEL"] = "INFO"

    # Setup logging (will read from environment)
    setup_logging(force=True)

    print("Log level set via LLM_CLIENT_LOG_LEVEL environment variable")

    client = LLMClient()
    print(f"Using provider: {client.api_choice}")


def example_disable_logging():
    """Example disabling all logging."""
    print("\n" + "=" * 60)
    print("Example 6: Disabling Logging")
    print("=" * 60)

    # First enable logging
    setup_logging(level="INFO", force=True)
    print("Logging enabled...")

    client = LLMClient()

    # Now disable it
    disable_logging()
    print("\nLogging disabled - no more log output below:")

    # These operations won't produce logs
    messages = [{"role": "user", "content": "Test"}]
    try:
        response = client.chat_completion(messages)
        print(f"Response received: {response[:50]}...")
    except Exception as e:
        print(f"Error: {e}")


def example_streaming_with_logging():
    """Example showing streaming with logging."""
    print("\n" + "=" * 60)
    print("Example 7: Streaming with Logging")
    print("=" * 60)

    # Enable DEBUG to see streaming details
    setup_logging(level="DEBUG", force=True)

    client = LLMClient()
    messages = [{"role": "user", "content": "Count from 1 to 5"}]

    print("\nStreaming response:")
    try:
        for chunk in client.chat_completion_stream(messages):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\nStreaming not supported: {e}")


def example_production_settings():
    """Example of production-appropriate logging settings."""
    print("\n" + "=" * 60)
    print("Example 8: Production Settings")
    print("=" * 60)

    # In production, use WARNING level (only errors and warnings)
    setup_logging(level="WARNING", force=True)

    print("Production mode: Only warnings and errors will be logged")

    client = LLMClient()
    messages = [{"role": "user", "content": "Hello"}]

    try:
        response = client.chat_completion(messages)
        print(f"Success: {response[:50]}...")
    except Exception as e:
        print(f"Error: {e}")


def example_conditional_logging():
    """Example of conditional logging based on environment."""
    print("\n" + "=" * 60)
    print("Example 9: Conditional Logging")
    print("=" * 60)

    # Check environment
    env = os.getenv("ENVIRONMENT", "development")
    print(f"Environment: {env}")

    # Configure logging based on environment
    if env == "development":
        setup_logging(level="DEBUG", force=True)
        print("Development mode: DEBUG logging enabled")
    elif env == "production":
        setup_logging(level="ERROR", force=True)
        print("Production mode: ERROR logging only")
    else:
        setup_logging(level="INFO", force=True)
        print("Default mode: INFO logging")

    LLMClient()


def example_error_logging():
    """Example showing error logging."""
    print("\n" + "=" * 60)
    print("Example 10: Error Logging")
    print("=" * 60)

    # Enable INFO logging
    setup_logging(level="INFO", force=True)

    try:
        # This will fail if API keys are not set
        client = LLMClient(api_choice="openai")
        client.switch_provider("invalid_provider")
    except Exception as e:
        print(f"\nCaught exception (as expected): {type(e).__name__}")
        print(f"Message: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LLM Client Logging Examples")
    print("=" * 60)
    print("\nThese examples demonstrate various logging configurations.")
    print("Some examples may fail if API keys are not configured.")
    print("\nYou can set log level via environment variable:")
    print("  export LLM_CLIENT_LOG_LEVEL=DEBUG")

    # Run examples
    examples = [
        example_basic_logging,
        example_debug_logging,
        example_provider_switching,
        example_custom_format,
        example_environment_variable,
        example_disable_logging,
        example_streaming_with_logging,
        example_production_settings,
        example_conditional_logging,
        example_error_logging,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nExample failed: {e}")
            print("This is expected if API keys are not configured.")

    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
