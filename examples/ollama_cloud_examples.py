"""Usage examples demonstrating Ollama Cloud functionality."""

from llm_client import LLMClient
from llm_client.exceptions import APIKeyNotFoundError, ChatCompletionError


def example_basic_cloud():
    """Basic Ollama Cloud usage."""
    print("=== Basic Ollama Cloud Usage ===\n")

    # Method 1: Explicit cloud mode
    client = LLMClient(api_choice="ollama", llm="gpt-oss:120b-cloud", use_ollama_cloud=True)

    messages = [{"role": "user", "content": "What is Ollama Cloud?"}]
    response = client.chat_completion(messages)
    print(f"Response: {response}\n")


def example_auto_detect_cloud():
    """Auto-detect cloud mode from model name."""
    print("=== Auto-Detect Cloud Mode ===\n")

    # Model name ending with '-cloud' auto-enables cloud mode
    client = LLMClient(llm="gpt-oss:120b-cloud")

    print(f"Using: {client.api_choice}")
    print(f"Model: {client.llm}")
    print(f"Cloud mode: {client.provider.use_cloud}\n")

    messages = [{"role": "user", "content": "Tell me about quantum computing."}]
    response = client.chat_completion(messages)
    print(f"Response: {response[:200]}...\n")


def example_streaming_cloud():
    """Streaming with Ollama Cloud."""
    print("=== Streaming with Ollama Cloud ===\n")

    client = LLMClient(llm="gpt-oss:120b-cloud")

    messages = [{"role": "user", "content": "Write a short poem about AI."}]

    print("Streaming response:")
    for chunk in client.chat_completion_stream(messages):
        print(chunk, end="", flush=True)
    print("\n")


def example_switch_local_cloud():
    """Switch between local and cloud Ollama."""
    print("=== Switch Between Local and Cloud ===\n")

    messages = [{"role": "user", "content": "What is 2+2?"}]

    # Start with local
    client = LLMClient(api_choice="ollama", llm="llama3.2:1b")
    print(f"Using local: {client.llm}")

    try:
        local_response = client.chat_completion(messages)
        print(f"Local response: {local_response}\n")
    except ChatCompletionError:
        print("Local Ollama not available\n")

    # Switch to cloud
    client.switch_provider("ollama", llm="gpt-oss:120b-cloud", use_ollama_cloud=True)
    print(f"Switched to cloud: {client.llm}")

    cloud_response = client.chat_completion(messages)
    print(f"Cloud response: {cloud_response}\n")


def example_config_file_cloud():
    """Load Ollama Cloud from config file."""
    print("=== Load from Config File ===\n")

    # Assuming llm_config.yaml has ollama_cloud provider
    try:
        client = LLMClient.from_config("llm_config.yaml", provider="ollama_cloud")

        print("Loaded from config:")
        print(f"  Provider: {client.api_choice}")
        print(f"  Model: {client.llm}")
        print(f"  Cloud mode: {client.provider.use_cloud}\n")

        messages = [{"role": "user", "content": "Hello from config!"}]
        response = client.chat_completion(messages)
        print(f"Response: {response[:100]}...\n")
    except FileNotFoundError:
        print("Config file not found\n")


def example_fallback_strategy():
    """Fallback from local to cloud."""
    print("=== Fallback Strategy: Local → Cloud ===\n")

    messages = [{"role": "user", "content": "Explain recursion."}]

    try:
        # Try local first
        print("Trying local Ollama...")
        client = LLMClient(api_choice="ollama", llm="llama3.2:1b")
        response = client.chat_completion(messages)
        print(f"Success with local: {response[:100]}...\n")

    except (ChatCompletionError, APIKeyNotFoundError):
        # Fallback to cloud
        print("Local unavailable, falling back to cloud...")
        client = LLMClient(llm="gpt-oss:120b-cloud")
        response = client.chat_completion(messages)
        print(f"Success with cloud: {response[:100]}...\n")


def example_hybrid_approach():
    """Use local for simple, cloud for complex tasks."""
    print("=== Hybrid Approach: Local for Simple, Cloud for Complex ===\n")

    # Local client for simple tasks
    local_client = LLMClient(api_choice="ollama", llm="llama3.2:1b")

    # Cloud client for complex tasks
    cloud_client = LLMClient(llm="gpt-oss:120b-cloud")

    # Simple task - use local
    simple_query = [{"role": "user", "content": "What is Python?"}]
    try:
        print("Simple task (local):")
        response = local_client.chat_completion(simple_query)
        print(f"  {response[:100]}...\n")
    except ChatCompletionError:
        print("  Local not available\n")

    # Complex task - use cloud
    complex_query = [
        {
            "role": "user",
            "content": "Provide a detailed analysis of the differences between "
            "procedural and object-oriented programming paradigms.",
        }
    ]
    print("Complex task (cloud):")
    response = cloud_client.chat_completion(complex_query)
    print(f"  {response[:150]}...\n")


def example_token_counting_cloud():
    """Token counting with cloud models."""
    print("=== Token Counting with Ollama Cloud ===\n")

    client = LLMClient(llm="gpt-oss:120b-cloud")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain machine learning in detail."},
    ]

    # Count tokens
    token_count = client.count_tokens(messages)
    print(f"Message tokens: {token_count}")

    # Check if within budget
    max_tokens = 4096
    reserved_for_response = 1000
    available = max_tokens - token_count - reserved_for_response

    print(f"Available for response: {available} tokens")

    if available > 0:
        response = client.chat_completion(messages)
        response_tokens = client.count_string_tokens(response)
        print(f"Response tokens: {response_tokens}")
        print(f"Total used: {token_count + response_tokens}\n")


def example_error_handling():
    """Proper error handling for Ollama Cloud."""
    print("=== Error Handling ===\n")

    try:
        # Try to create client without API key
        client = LLMClient(api_choice="ollama", llm="gpt-oss:120b-cloud", use_ollama_cloud=True)

        messages = [{"role": "user", "content": "Test"}]
        response = client.chat_completion(messages)
        print(f"Success: {response[:50]}...\n")

    except APIKeyNotFoundError as e:
        print(f"API Key Error: {e}")
        print("Solution: Set OLLAMA_API_KEY environment variable\n")

    except ChatCompletionError as e:
        print(f"Chat Error: {e}")
        print("Check your internet connection and API key\n")


def example_custom_host():
    """Use custom Ollama host."""
    print("=== Custom Ollama Host ===\n")

    # Custom cloud host
    client = LLMClient(
        api_choice="ollama",
        llm="gpt-oss:120b-cloud",
        use_ollama_cloud=True,
        ollama_host="https://ollama.com",
    )

    print(f"Using host: {client.ollama_host}")
    print(f"Model: {client.llm}\n")

    messages = [{"role": "user", "content": "Hello!"}]
    response = client.chat_completion(messages)
    print(f"Response: {response}\n")


def example_comparison():
    """Compare local vs cloud responses."""
    print("=== Compare Local vs Cloud ===\n")

    messages = [{"role": "user", "content": "What is artificial intelligence?"}]

    # Local response
    try:
        print("Local Ollama (llama3.2:1b):")
        local_client = LLMClient(api_choice="ollama", llm="llama3.2:1b")
        local_response = local_client.chat_completion(messages)
        print(f"  {local_response[:150]}...\n")
    except ChatCompletionError:
        print("  Local not available\n")

    # Cloud response
    print("Ollama Cloud (gpt-oss:120b-cloud):")
    cloud_client = LLMClient(llm="gpt-oss:120b-cloud")
    cloud_response = cloud_client.chat_completion(messages)
    print(f"  {cloud_response[:150]}...\n")


if __name__ == "__main__":
    """Run all examples."""

    print("\n" + "=" * 60)
    print("OLLAMA CLOUD EXAMPLES")
    print("=" * 60 + "\n")

    try:
        example_basic_cloud()
        example_auto_detect_cloud()
        example_streaming_cloud()
        example_switch_local_cloud()
        example_config_file_cloud()
        example_fallback_strategy()
        example_hybrid_approach()
        example_token_counting_cloud()
        example_error_handling()
        example_custom_host()
        example_comparison()

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nMake sure to:")
        print("1. Set OLLAMA_API_KEY environment variable")
        print("2. Install: pip install llm_client ollama")
        print("3. For local: Install Ollama from https://ollama.ai")

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60 + "\n")
