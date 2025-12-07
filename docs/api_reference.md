# API Reference

## LLMClient

The main client class for interacting with LLM providers.

### Constructor

```python
LLMClient(
    llm: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    api_choice: Literal["openai", "groq", "gemini", "ollama"] | None = None,
    secrets_path: str = "secrets.env",
    keep_alive: str = "5m",
    use_async: bool = False
)
```

**Parameters:**

- `llm` (str, optional): Model name. If None, uses provider's default model.
- `temperature` (float): Sampling temperature (0.0 to 2.0). Default: 0.7
- `max_tokens` (int): Maximum tokens to generate. Default: 512
- `api_choice` (str, optional): Explicitly select provider. If None, auto-selects.
- `secrets_path` (str): Path to secrets file. Default: "secrets.env"
- `keep_alive` (str): Ollama keep-alive duration. Default: "5m"
- `use_async` (bool): Use async providers. Default: False

**Returns:** LLMClient instance

**Example:**

```python
# Auto-select with defaults
client = LLMClient()

# Custom configuration
client = LLMClient(
    api_choice="openai",
    llm="gpt-4o",
    temperature=0.5,
    max_tokens=2048
)

# Async client
async_client = LLMClient(use_async=True)
```

---

### from_config

```python
@classmethod
LLMClient.from_config(
    config_path: str | Path,
    provider: str | None = None,
    secrets_path: str = "secrets.env",
    use_async: bool = False
) -> LLMClient
```

Create client from YAML or JSON configuration file.

**Parameters:**

- `config_path` (str | Path): Path to configuration file
- `provider` (str, optional): Provider to use. If None, uses default from config
- `secrets_path` (str): Path to secrets file
- `use_async` (bool): Create async client

**Returns:** Configured LLMClient instance

**Raises:**
- `FileNotFoundError`: Config file doesn't exist
- `ValueError`: Invalid configuration

**Example:**

```python
# Load default provider
client = LLMClient.from_config("llm_config.yaml")

# Load specific provider
client = LLMClient.from_config("llm_config.yaml", provider="groq")

# Async client from config
client = LLMClient.from_config("config.yaml", use_async=True)
```

---

### chat_completion

```python
chat_completion(
    messages: list[dict[str, str]]
) -> str
```

Execute chat completion with automatic retry logic.

**Parameters:**

- `messages` (list[dict]): List of message dictionaries with 'role' and 'content' keys

**Returns:** str - Generated text response

**Raises:**
- `ChatCompletionError`: API call failed after all retries

**Example:**

```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is AI?"}
]

response = client.chat_completion(messages)
```

---

### chat_completion_stream

```python
chat_completion_stream(
    messages: list[dict[str, str]]
) -> Iterator[str]
```

Stream response tokens as they arrive.

**Parameters:**

- `messages` (list[dict]): List of message dictionaries

**Returns:** Iterator[str] - Iterator yielding response chunks

**Raises:**
- `StreamingNotSupportedError`: Provider doesn't support streaming
- `ChatCompletionError`: API call failed

**Example:**

```python
messages = [{"role": "user", "content": "Tell me a story"}]

for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
print()  # New line after streaming
```

---

### chat_completion_with_tools

```python
chat_completion_with_tools(
    messages: list[dict[str, str]],
    tools: list[dict],
    tool_choice: str | dict | None = None
) -> dict
```

Execute chat completion with function/tool calling.

**Parameters:**

- `messages` (list[dict]): List of message dictionaries
- `tools` (list[dict]): Tool definitions in OpenAI format
- `tool_choice` (str | dict, optional): Controls tool selection
  - `"auto"`: LLM decides (default)
  - `"none"`: No tools called
  - `{"type": "function", "function": {"name": "..."}`: Force specific tool

**Returns:** dict with keys:
- `content` (str | None): Generated text
- `tool_calls` (list | None): List of tool calls made

**Raises:**
- `NotImplementedError`: Provider doesn't support tools
- `ChatCompletionError`: API call failed

**Example:**

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
}]

result = client.chat_completion_with_tools(messages, tools)

if result['tool_calls']:
    for call in result['tool_calls']:
        print(f"Function: {call['function']['name']}")
        print(f"Args: {call['function']['arguments']}")
```

---

### switch_provider

```python
switch_provider(
    api_choice: Literal["openai", "groq", "gemini", "ollama"],
    llm: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None
) -> None
```

Switch to different provider at runtime.

**Parameters:**

- `api_choice` (str): Target provider
- `llm` (str, optional): New model name. If None, uses provider default
- `temperature` (float, optional): New temperature. If None, keeps current
- `max_tokens` (int, optional): New max_tokens. If None, keeps current

**Raises:**
- `InvalidProviderError`: Invalid provider name
- `APIKeyNotFoundError`: API key missing

**Example:**

```python
# Switch provider
client.switch_provider("gemini")

# Switch with new model
client.switch_provider("openai", llm="gpt-4o")

# Switch with new parameters
client.switch_provider("groq", temperature=0.3, max_tokens=1024)
```

---

### count_tokens

```python
count_tokens(
    messages: list[dict[str, str]],
    model: str | None = None
) -> int
```

Count tokens in messages using tiktoken.

**Parameters:**

- `messages` (list[dict]): Messages to count tokens for
- `model` (str, optional): Model name. If None, uses current model

**Returns:** int - Total token count

**Example:**

```python
messages = [{"role": "user", "content": "Hello world"}]
token_count = client.count_tokens(messages)

# With different model
count = client.count_tokens(messages, model="gpt-3.5-turbo")
```

---

### count_string_tokens

```python
count_string_tokens(
    text: str,
    model: str | None = None
) -> int
```

Count tokens in a string.

**Parameters:**

- `text` (str): Text to count tokens for
- `model` (str, optional): Model name. If None, uses current model

**Returns:** int - Token count

**Example:**

```python
token_count = client.count_string_tokens("Hello, world!")
```

---

### Async Methods

All async methods require `use_async=True` when creating the client.

#### achat_completion

```python
async achat_completion(
    messages: list[dict[str, str]]
) -> str
```

Async version of chat_completion.

**Example:**

```python
client = LLMClient(use_async=True)
response = await client.achat_completion(messages)
```

#### achat_completion_stream

```python
async achat_completion_stream(
    messages: list[dict[str, str]]
) -> AsyncIterator[str]
```

Async streaming version.

**Example:**

```python
async for chunk in client.achat_completion_stream(messages):
    print(chunk, end="", flush=True)
```

#### achat_completion_with_tools

```python
async achat_completion_with_tools(
    messages: list[dict[str, str]],
    tools: list[dict],
    tool_choice: str | dict | None = None
) -> dict
```

Async tool calling.

---

### Properties

#### llm

```python
@property
llm -> str
```

Current model name.

```python
print(client.llm)  # "gpt-4o-mini"
```

#### client

```python
@property
client -> Any
```

Underlying provider client (for backward compatibility).

---

## LLMConfig

Configuration file loader.

### Constructor

```python
LLMConfig(config_dict: dict[str, Any])
```

**Parameters:**

- `config_dict` (dict): Configuration dictionary

---

### from_file

```python
@classmethod
LLMConfig.from_file(file_path: str | Path) -> LLMConfig
```

Load configuration from YAML or JSON file.

**Parameters:**

- `file_path` (str | Path): Path to config file (.yaml, .yml, or .json)

**Returns:** LLMConfig instance

**Raises:**
- `FileNotFoundError`: File doesn't exist
- `ValueError`: Unsupported format or invalid config
- `ImportError`: YAML file but pyyaml not installed

**Example:**

```python
config = LLMConfig.from_file("llm_config.yaml")
```

---

### from_dict

```python
@classmethod
LLMConfig.from_dict(config_dict: dict[str, Any]) -> LLMConfig
```

Create configuration from dictionary.

**Example:**

```python
config_dict = {
    "default_provider": "openai",
    "providers": {
        "openai": {"model": "gpt-4o"}
    }
}
config = LLMConfig.from_dict(config_dict)
```

---

### validate

```python
validate() -> tuple[bool, list[str]]
```

Validate configuration.

**Returns:** Tuple of (is_valid, error_messages)

**Example:**

```python
is_valid, errors = config.validate()
if not is_valid:
    print(f"Errors: {errors}")
```

---

## TokenCounter

Utility for counting tokens.

### count_tokens

```python
@staticmethod
TokenCounter.count_tokens(
    messages: list[dict[str, str]],
    model: str = "gpt-4o-mini",
    fallback: bool = True
) -> int
```

Count tokens in messages.

**Parameters:**

- `messages` (list[dict]): Messages to count
- `model` (str): Model name for encoding
- `fallback` (bool): Use estimation if tiktoken unavailable

**Returns:** int - Token count

**Raises:**
- `ImportError`: tiktoken not installed and fallback=False

---

### count_string_tokens

```python
@staticmethod
TokenCounter.count_string_tokens(
    text: str,
    model: str = "gpt-4o-mini"
) -> int
```

Count tokens in a string.

---

### is_tiktoken_available

```python
@staticmethod
TokenCounter.is_tiktoken_available() -> bool
```

Check if tiktoken is installed.

---

## Exceptions

### LLMClientError

Base exception for all LLM client errors.

```python
from llm_client.exceptions import LLMClientError

try:
    client = LLMClient()
except LLMClientError as e:
    print(f"LLM Client error: {e}")
```

---

### APIKeyNotFoundError

Raised when required API key is missing.

**Attributes:**
- `provider` (str): Provider name
- `key_name` (str): Required environment variable name

---

### ProviderNotAvailableError

Raised when provider package is not installed.

**Attributes:**
- `provider` (str): Provider name
- `package_name` (str): Required pip package

---

### InvalidProviderError

Raised when invalid provider name is specified.

**Attributes:**
- `provider` (str): Invalid provider name
- `valid_providers` (list[str]): List of valid providers

---

### ChatCompletionError

Raised when chat completion fails.

**Attributes:**
- `provider` (str): Provider name
- `original_error` (Exception): Original exception

---

### StreamingNotSupportedError

Raised when streaming is not supported.

**Attributes:**
- `provider` (str): Provider name
- `reason` (str | None): Optional reason

---

## Utility Functions

### generate_config_template

```python
from llm_client.config import generate_config_template

generate_config_template(
    output_path: str | Path,
    format: str = "yaml"
) -> None
```

Generate a template configuration file.

**Parameters:**
- `output_path` (str | Path): Where to save template
- `format` (str): "yaml" or "json"

**Example:**

```python
generate_config_template("llm_config.yaml", format="yaml")
```

---

### create_default_config

```python
from llm_client.config import create_default_config

create_default_config() -> dict[str, Any]
```

Create default configuration dictionary.

**Returns:** dict - Default configuration

**Example:**

```python
config_dict = create_default_config()
config = LLMConfig.from_dict(config_dict)
```
