# Installation

Learn how to install the LLM Client and set it up for development.

## Quick Install

You can install the package directly from GitHub using pip:

```bash
pip install git+https://github.com/dgaida/llm_client.git
```

## Development Install

For local development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/dgaida/llm_client.git
cd llm_client
pip install -e ".[dev]"
```

## Optional Dependencies

The LLM Client offers various extras for additional functionality:

```bash
# With LlamaIndex support
pip install -e ".[llama-index]"

# With all features and development tools
pip install -e ".[all]"
```

## System Requirements

- **Python**: 3.10 or higher  
- **OS**: Windows, macOS, or Linux  
- **Internet Connection**: Required for cloud providers (OpenAI, Groq, Gemini)  
- **Ollama**: Required for local model usage  
