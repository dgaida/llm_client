# LLM Client CLI Usage Guide

Command-line interface for quick LLM access across multiple providers.

## Installation

```bash
# Basic installation
pip install git+https://github.com/dgaida/llm_client.git

# With rich formatting support (recommended)
pip install git+https://github.com/dgaida/llm_client.git[rich]

# Development installation
git clone https://github.com/dgaida/llm_client.git
cd llm_client
pip install -e ".[all]"
```

After installation, you can use either:
- `llm-client` (full command)
- `llm` (shorter alias)


## Quick Start

```bash
# Simple query
llm-client chat "What is Python?"

# Interactive mode
llm-client interactive

# With specific provider
llm-client chat "Explain AI" --provider openai

# Stream response
llm-client chat "Tell me a story" --stream

# Count tokens
llm-client tokens "Hello world!"
```

---

## Commands

### `chat` - One-Shot Queries

Execute a single chat completion and exit.

**Usage:**
```bash
llm-client chat [OPTIONS] PROMPT
```

**Options:**
- `-p, --provider [auto|openai|groq|gemini|ollama]` - Provider to use (default: auto)
- `-m, --model TEXT` - Specific model name
- `-t, --temperature FLOAT` - Sampling temperature (0.0-2.0, default: 0.7)
- `--max-tokens INTEGER` - Maximum tokens to generate (default: 512)
- `--stream / --no-stream` - Enable streaming (default: no-stream)
- `-c, --config PATH` - Load from config file
- `--markdown / --no-markdown` - Render as markdown (default: markdown)

**Examples:**
```bash
# Basic query
llm-client chat "What is machine learning?"

# With specific provider and model
llm-client chat "Explain quantum computing" --provider openai --model gpt-4o

# Stream response in real-time
llm-client chat "Tell me a long story about AI" --stream

# Lower temperature for more focused responses
llm-client chat "What is 2+2?" --temperature 0.1

# From config file
llm-client chat "Hello" --config llm_config.yaml

# Use Ollama Cloud
llm-client chat "Analyze this code" --provider ollama --model gpt-oss:120b-cloud
```

---

### `interactive` - Chat Sessions

Start an interactive chat session with conversation history.

**Usage:**
```bash
llm-client interactive [OPTIONS]
```

**Options:**
- `-p, --provider [auto|openai|groq|gemini|ollama]` - Provider to use
- `-m, --model TEXT` - Specific model name
- `-t, --temperature FLOAT` - Sampling temperature (default: 0.7)
- `-c, --config PATH` - Load from config file
- `-s, --system TEXT` - System message for context

**Interactive Commands:**
- `exit` or `quit` - Exit the session
- `clear` - Clear conversation history
- `switch <provider>` - Switch to different provider
- `Ctrl+C` or `Ctrl+D` - Exit gracefully

**Examples:**
```bash
# Start interactive session
llm-client interactive

# With specific provider
llm-client interactive --provider openai

# With system message (set context)
llm-client interactive --system "You are a Python expert who gives concise answers"

# With config file
llm-client interactive --config llm_config.yaml --provider groq

# Coding assistant
llm-client interactive --system "You are an expert software engineer" --provider openai
```

**Sample Session:**
```
╭─────────────────────────────────────────╮
│     Interactive Chat Mode              │
│ Provider: openai                        │
│ Model: gpt-4o-mini                      │
│ Temperature: 0.7                        │
│                                         │
│ Commands: exit, quit, clear, switch    │
╰─────────────────────────────────────────╯

You: What is Python?

---

## Configuration

The CLI automatically loads API keys from:
1. Environment variables
2. `.env` file in the current directory
3. `secrets.env` file

You can also specify a custom configuration file using the `--config` option.

## Examples

### Multimodal Query (Image Analysis)
```bash
llm-client chat "What is in this image?" --provider gemini --file image.jpg
```

### Complex System Prompt
```bash
llm-client chat "Write a unit test for this function" --system "You are a senior QA engineer"
```
