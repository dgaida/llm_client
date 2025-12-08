"""
examples
========

Example scripts demonstrating llm_client usage.

Available examples:
- usage_examples.py: Token counting, async support, config files
- streaming_example.py: Response streaming and retry logic
- ollama_cloud_examples.py: Ollama Cloud integration examples

Quick Start:
    from llm_client import LLMClient

    # Automatic API detection
    client = LLMClient()

    messages = [{"role": "user", "content": "Hello!"}]
    response = client.chat_completion(messages)
    print(response)

Features demonstrated:
- Token counting with tiktoken
- Async/await operations
- Configuration file management
- Response streaming
- Provider switching
- Ollama Cloud vs Local
- Error handling with custom exceptions

For more examples, see:
- notebooks/llm_client_example.ipynb
- notebooks/RAGChatbot_groq_API.ipynb
"""

# Empty file - examples are standalone scripts
# This allows examples directory to be imported as a package if needed
