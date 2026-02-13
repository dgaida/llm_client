# Troubleshooting

Solutions for common issues with the LLM Client.

## Table of Contents

- [Installation](#installation)
- [API Keys](#api-keys)
- [Provider Issues](#provider-issues)
- [Ollama](#ollama)
- [Streaming](#streaming)
- [Token Counting](#token-counting)
- [Async Issues](#async-issues)
- [Configuration Files](#configuration-files)
- [File Upload](#file-upload)
- [Performance](#performance)

---

## Installation

### ImportError: No module named 'llm_client'

**Problem**: After installation, llm_client cannot be imported.

**Solution**:
```bash
# Check if installation was successful
pip list | grep llm-client

# Reinstall
pip uninstall llm-client
pip install git+https://github.com/dgaida/llm_client.git

# Or use editable install
git clone https://github.com/dgaida/llm_client.git
cd llm_client
pip install -e .
```

### ModuleNotFoundError: No module named 'openai'/'groq'/'ollama'

**Problem**: Provider packages are missing.

**Solution**:
```bash
# For OpenAI/Gemini
pip install openai

# For Groq
pip install groq

# For Ollama
pip install ollama

# Or all at once
pip install openai groq ollama
```

### ImportError: tiktoken required

**Problem**: tiktoken not installed, but needed for token counting.

**Solution**:
```bash
pip install tiktoken

# Or all optional dependencies
pip install -e ".[all]"
```

---

## API Keys

### APIKeyNotFoundError

**Problem**: API key not found.

```python
APIKeyNotFoundError: OPENAI_API_KEY not found for openai provider.
```

**Solution 1 - Environment Variable**:
```bash
# Linux/macOS
export OPENAI_API_KEY=sk-...

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."

# Windows (CMD)
set OPENAI_API_KEY=sk-...
```

**Solution 2 - secrets.env File**:
```bash
# Create secrets.env in project directory
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk-...
GEMINI_API_KEY=AIzaSy-...
```

**Solution 3 - Google Colab**:
```python
# Add keys in Colab Secrets (🔑 icon in left sidebar)
# Key Name: OPENAI_API_KEY
# Value: sk-...

from llm_client import LLMClient
client = LLMClient()  # Automatically loads from Colab Secrets
```

**Solution 4 - Programmatically**:
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

from llm_client import LLMClient
client = LLMClient()
```

### API Key not recognized

**Problem**: Key is set but not recognized.

**Verification**:
```python
import os

# Print key (safe in local environment)
print(os.getenv("OPENAI_API_KEY"))

# In Colab
from google.colab import userdata
print(userdata.get("OPENAI_API_KEY"))
```

**Possible Causes**:
- Spaces in the key
- Incorrect key name (e.g., `OPENAI_KEY` instead of `OPENAI_API_KEY`)
- Key in incorrect format

---

## Provider Issues

### InvalidProviderError

**Problem**: Invalid provider name.

```python
InvalidProviderError: Invalid provider: opena1. Valid providers are: openai, groq, gemini, ollama
```

**Solution**:
```python
# Correct provider names
client = LLMClient(api_choice="openai")   # ✓
client = LLMClient(api_choice="groq")     # ✓
client = LLMClient(api_choice="gemini")   # ✓
client = LLMClient(api_choice="ollama")   # ✓

# List available providers
from llm_client import ProviderFactory
available = ProviderFactory.get_available_providers()
print(f"Available: {available}")
```

### ProviderNotAvailableError

**Problem**: Provider package not installed.

```python
ProviderNotAvailableError: groq provider not available. Install with: pip install groq
```

**Solution**:
```bash
# Install missing package
pip install groq

# Or all provider packages
pip install openai groq ollama
```

### Provider Switch Fails

**Problem**: `switch_provider()` leads to error.

**Solution**:
```python
from llm_client.exceptions import APIKeyNotFoundError

try:
    client.switch_provider("groq")
except APIKeyNotFoundError as e:
    print(f"API Key missing: {e.key_name}")
    # Set key and try again
    import os
    os.environ["GROQ_API_KEY"] = "gsk-..."
    client.switch_provider("groq")
```

---

## Ollama

### Ollama not running

**Problem**: Connection to local Ollama fails.

```python
ChatCompletionError: Connection refused
```

**Solution**:
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
# macOS/Linux
ollama serve

# Systemd (Linux)
sudo systemctl start ollama
sudo systemctl status ollama
```

### Model not found

**Problem**: Model not available.

```python
ChatCompletionError: model 'llama3.2:1b' not found
```

**Solution**:
```bash
# List installed models
ollama list

# Pull missing model
ollama pull llama3.2:1b

# Popular models
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull mixtral:8x7b
```

### Out of Memory (Ollama local)

**Problem**: Model is too large for available RAM.

**Solution**:
```python
# Use a smaller model
client = LLMClient(api_choice="ollama", llm="llama3.2:1b")  # ~1.3GB

# Or a quantized model
client = LLMClient(api_choice="ollama", llm="llama3.1:8b-q4_0")  # Smaller
```

**Recommendations by RAM**:
- < 4GB RAM: `llama3.2:1b`
- 4-8GB RAM: `llama3.2:3b`
- 8-16GB RAM: `llama3.1:8b`
- 16GB+ RAM: `llama3.1:70b` or larger

### Ollama Cloud API Key missing

**Problem**: Ollama Cloud without API key.

```python
APIKeyNotFoundError: OLLAMA_API_KEY not found for ollama_cloud provider
```

**Solution**:
```bash
# Set Ollama Cloud API Key
export OLLAMA_API_KEY=your_api_key

# Or in secrets.env
echo "OLLAMA_API_KEY=your_api_key" >> secrets.env
```

---

## Streaming

### StreamingNotSupportedError

**Problem**: Streaming is not supported.

**Solution**:
```python
from llm_client.exceptions import StreamingNotSupportedError

try:
    for chunk in client.chat_completion_stream(messages):
        print(chunk, end="")
except StreamingNotSupportedError:
    # Fallback to normal completion
    response = client.chat_completion(messages)
    print(response)
```

### Streaming Freezes

**Problem**: Stream stops in the middle of a response.

**Possible Causes**:
- Network issues
- Timeout
- API limit reached

**Solution**:
```python
import time

# With timeout handling
chunks = []
try:
    for chunk in client.chat_completion_stream(messages):
        chunks.append(chunk)
        print(chunk, end="", flush=True)
except Exception as e:
    print(f"\nStreaming interrupted: {e}")
    if chunks:
        print(f"Partial response: {''.join(chunks)}")
```

---

## Token Counting

### tiktoken not available

**Problem**: Token counting without tiktoken.

**Solution**:
```bash
# Install tiktoken
pip install tiktoken
```

**Workaround without tiktoken**:
```python
from llm_client import TokenCounter

# Automatically uses estimation
counter = TokenCounter()
token_count = counter.count_tokens(messages)
print(f"Estimated tokens: {token_count}")
```

### Incorrect Token Count

**Problem**: Token count deviates significantly.

**Causes**:
- Incorrect model for encoding
- tiktoken not installed (estimation)

**Solution**:
```python
# Specify correct model
token_count = client.count_tokens(messages, model="gpt-4o")

# For specific models
from llm_client import TokenCounter
counter = TokenCounter()

# GPT-4o
count = counter.count_tokens(messages, model="gpt-4o")

# GPT-3.5
count = counter.count_tokens(messages, model="gpt-3.5-turbo")
```

---

## Async Issues

### RuntimeError: Async methods not supported

**Problem**: Async methods called on a Sync client.

```python
RuntimeError: AsyncOpenAIProvider does not support async methods
```

**Solution**:
```python
# Create Async client
client = LLMClient(use_async=True)

# Use async methods
import asyncio

async def main():
    response = await client.achat_completion(messages)
    print(response)

asyncio.run(main())
```

### Event loop is already running

**Problem**: In Jupyter/Colab.

**Solution**:
```python
# In Jupyter/Colab
import nest_asyncio
nest_asyncio.apply()

# Then as usual
asyncio.run(main())
```

### Async Provider not available

**Problem**: Import error for async_providers.

**Solution**:
```bash
# Install async dependencies
pip install asyncio

# Or full installation
pip install -e ".[all]"
```

---

## Configuration Files

### FileNotFoundError: Config file not found

**Problem**: Config file not found.

**Solution**:
```python
from pathlib import Path

# Check path
config_path = Path("llm_config.yaml")
if not config_path.exists():
    print(f"File not found: {config_path.absolute()}")

    # Create template
    from llm_client.config import generate_config_template
    generate_config_template("llm_config.yaml")
```

### ValueError: Invalid configuration

**Problem**: Configuration invalid.

**Solution**:
```python
from llm_client.config import LLMConfig

# Validate config
config = LLMConfig.from_file("llm_config.yaml")
is_valid, errors = config.validate()

if not is_valid:
    print("Errors in configuration:")
    for error in errors:
        print(f"  - {error}")
```

**Common Errors**:
- Missing `model` parameter
- Invalid `default_provider`
- YAML syntax error

### ImportError: pyyaml required

**Problem**: YAML file used but pyyaml missing.

**Solution**:
```bash
pip install pyyaml
```

**Alternative - Use JSON**:
```python
# Use JSON instead of YAML
generate_config_template("llm_config.json", format="json")
```

---

## File Upload

### FileNotFoundError: File not found

**Problem**: File does not exist.

**Solution**:
```python
from pathlib import Path

files = ["image.jpg", "document.pdf"]

# Check files
for file_path in files:
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        print(f"Current directory: {Path.cwd()}")
```

### ValueError: Unsupported file type

**Problem**: File type not supported.

**Solution**:
```python
from llm_client.file_utils import validate_file_for_provider

# Check file for provider
is_valid, error = validate_file_for_provider("video.mp4", "openai")

if not is_valid:
    print(f"Error: {error}")
    # OpenAI does not support videos
    # Use Gemini instead
    client = LLMClient(api_choice="gemini")
```

**Provider File Support**:
- OpenAI: Images, PDFs
- Gemini: Images, PDFs, Videos, Audio
- Groq: Images only (Vision models)
- Ollama: Images only (Vision models)

### FileUploadNotSupportedError

**Problem**: Provider does not support file uploads.

**Solution**:
```python
from llm_client.exceptions import FileUploadNotSupportedError

try:
    response = client.chat_completion_with_files(messages, files=["image.jpg"])
except FileUploadNotSupportedError as e:
    print(f"File upload not supported: {e}")
    # Switch to provider with file support
    client.switch_provider("gemini")
    response = client.chat_completion_with_files(messages, files=["image.jpg"])
```

---

## Performance

### Slow Responses

**Problem**: API calls take a very long time.

**Possible Causes & Solutions**:

**1. Network Issues**:
```python
import time

start = time.time()
response = client.chat_completion(messages)
elapsed = time.time() - start

print(f"Duration: {elapsed:.2f}s")

# If > 10s: Network issue or API overload
```

**2. Too many tokens**:
```python
# Check token count
token_count = client.count_tokens(messages)
print(f"Tokens: {token_count}")

# Reduce input
if token_count > 2000:
    # Shorten messages or use summarization
    pass
```

**3. Incorrect Model**:
```python
# Use faster models
client = LLMClient(api_choice="groq")  # Very fast
# or
client = LLMClient(api_choice="openai", llm="gpt-4o-mini")  # Faster than gpt-4o
```

### Rate Limit Errors

**Problem**: Too many requests.

```python
ChatCompletionError: Rate limit exceeded
```

**Solution**:
```python
import time

# Simple retry with delay
for attempt in range(3):
    try:
        response = client.chat_completion(messages)
        break
    except Exception as e:
        if "rate" in str(e).lower():
            wait_time = (attempt + 1) * 5
            print(f"Rate limit, waiting {wait_time}s...")
            time.sleep(wait_time)
        else:
            raise

# Or use built-in retry (automatic)
response = client.chat_completion(messages)  # Retry is already built-in
```

### Memory Errors (local Ollama)

**Problem**: Out of Memory with large models.

**Solution**:
```python
# 1. Use a smaller model
client = LLMClient(api_choice="ollama", llm="llama3.2:1b")

# 2. Unload model after use
client = LLMClient(api_choice="ollama", keep_alive="0")

# 3. Quantized model
client = LLMClient(api_choice="ollama", llm="llama3.1:8b-q4_0")

# 4. Use Ollama Cloud (no local GPU needed)
client = LLMClient(llm="gpt-oss:120b-cloud")
```

---

## Further Help

### Enable Debug Logging

```python
from llm_client import enable_logging

# Enable debug logging
enable_logging("DEBUG")

# Now all internal steps are logged
client = LLMClient()
response = client.chat_completion(messages)
```

### Detailed Error Info

```python
from llm_client.exceptions import ChatCompletionError

try:
    response = client.chat_completion(messages)
except ChatCompletionError as e:
    print(f"Provider: {e.provider}")
    print(f"Original Error: {e.original_error}")
    print(f"Error Type: {type(e.original_error).__name__}")
```

### Create Issue on GitHub

If the problem persists:

1. Visit [GitHub Issues](https://github.com/dgaida/llm_client/issues)
2. Check if the problem has already been reported
3. Create a new issue with:
   - Python version (`python --version`)
   - Operating System
   - LLM Client version
   - Provider used
   - Minimal reproduction example
   - Full error traceback

### Support Channels

- [GitHub Issues](https://github.com/dgaida/llm_client/issues) - Bug Reports & Feature Requests
- [GitHub Discussions](https://github.com/dgaida/llm_client/discussions) - Questions & Discussions
- [Documentation](https://github.com/dgaida/llm_client/tree/master/docs) - Full documentation
