# Logging in llm_client

The `llm_client` package includes comprehensive logging support to help with debugging, monitoring, and understanding the behavior of your LLM applications.

## Quick Start

### Enable Logging

```python
from llm_client import LLMClient, setup_logging

# Enable INFO level logging
setup_logging(level="INFO")

# Create client and use normally
client = LLMClient()
messages = [{"role": "user", "content": "Hello!"}]
response = client.chat_completion(messages)
```

### Default Behavior

By default, logging is set to `WARNING` level. This means only warnings and errors are logged. To see more detailed information, enable `INFO` or `DEBUG` logging.

## Configuration

### Via setup_logging()

```python
from llm_client import setup_logging

# Set to DEBUG for maximum verbosity
setup_logging(level="DEBUG")

# Set to INFO for moderate verbosity
setup_logging(level="INFO")

# Set to WARNING (default)
setup_logging(level="WARNING")

# Set to ERROR to only see errors
setup_logging(level="ERROR")

# Custom format
setup_logging(
    level="INFO",
    format_string="%(levelname)s - %(name)s - %(message)s"
)
```

### Via Environment Variable

```bash
# Set before running your script
export LLM_CLIENT_LOG_LEVEL=DEBUG
python your_script.py
```

```python
# In your script
from llm_client import LLMClient

# Automatically uses LLM_CLIENT_LOG_LEVEL from environment
client = LLMClient()
```

### Enable/Disable Logging

```python
from llm_client import enable_logging, disable_logging

# Enable at INFO level
enable_logging("INFO")

# Do some work...
client = LLMClient()

# Disable all logging
disable_logging()
```

## Log Levels

### DEBUG
Most verbose. Shows:
- Provider initialization details
- API keys found (without exposing values)
- Model selection process
- Token counting operations
- Every API call with parameters
- Response sizes and chunk counts

**Use when:** Debugging issues, understanding internal behavior

```python
setup_logging(level="DEBUG")
```

Example output:
```
2024-12-08 10:30:15 - llm_client.llm_client - DEBUG - Initializing LLMClient with api_choice=None, llm=None
2024-12-08 10:30:15 - llm_client.llm_client - DEBUG - Loading secrets from secrets.env
2024-12-08 10:30:15 - llm_client.llm_client - DEBUG - Found API keys for: OpenAI, Groq
2024-12-08 10:30:15 - llm_client.provider_factory - DEBUG - Creating provider with api_choice=None, async=False
2024-12-08 10:30:15 - llm_client.provider_factory - DEBUG - Auto-selecting API based on available keys
2024-12-08 10:30:15 - llm_client.provider_factory - DEBUG - Selected OpenAI (API key found)
```

### INFO
Moderate verbosity. Shows:
- Provider creation and initialization
- Provider switching
- Model and API being used
- High-level operation status

**Use when:** Monitoring application flow, tracking provider usage

```python
setup_logging(level="INFO")
```

Example output:
```
2024-12-08 10:30:15 - llm_client.llm_client - INFO - Creating provider for API: auto-detect
2024-12-08 10:30:15 - llm_client.provider_factory - INFO - Auto-selected API: openai
2024-12-08 10:30:15 - llm_client.providers - INFO - OpenAI client initialized with model gpt-4o-mini
2024-12-08 10:30:15 - llm_client.llm_client - INFO - Initialized with provider: openai, model: gpt-4o-mini
```

### WARNING (Default)
Shows only warnings and errors. Minimal noise.

**Use when:** Production deployments, normal operation

### ERROR
Only shows errors. Very quiet.

**Use when:** You only care about failures

### CRITICAL
Only critical errors. Almost silent.

## What Gets Logged

### Client Initialization
- API key availability (without exposing keys)
- Provider selection process
- Model configuration
- Google Colab secret loading attempts

### Provider Operations
- API calls with model name and message count
- Response sizes
- Streaming chunk counts
- Tool calling operations

### Provider Switching
- Old and new provider information
- Configuration changes
- Reason for switching

### Token Counting
- Number of tokens counted
- Model used for counting
- Number of messages processed

### Errors
- Missing API keys with helpful messages
- Provider initialization failures
- API call failures with context

## Examples

### Basic Usage with Logging

```python
from llm_client import LLMClient, setup_logging

# Enable INFO logging
setup_logging(level="INFO")

# Create client (will log provider selection)
client = LLMClient()

# Make a request (will log API call)
messages = [{"role": "user", "content": "Hello!"}]
response = client.chat_completion(messages)
```

### Debugging Issues

```python
from llm_client import LLMClient, setup_logging

# Enable DEBUG logging to see everything
setup_logging(level="DEBUG")

# Now you can see exactly what's happening
client = LLMClient(api_choice="openai", llm="gpt-4o")
response = client.chat_completion(messages)
```

### Production Usage

```python
from llm_client import LLMClient

# Default WARNING level is fine for production
client = LLMClient()

# Only errors and warnings will be logged
response = client.chat_completion(messages)
```

### Conditional Logging

```python
import os
from llm_client import LLMClient, setup_logging

# Enable debug logging in development
if os.getenv("ENVIRONMENT") == "development":
    setup_logging(level="DEBUG")
else:
    setup_logging(level="WARNING")

client = LLMClient()
```

### Custom Log Format

```python
from llm_client import setup_logging

# Simple format
setup_logging(
    level="INFO",
    format_string="%(levelname)s: %(message)s"
)

# Detailed format with timestamps
setup_logging(
    level="DEBUG",
    format_string="[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
)
```

### Logging to File

```python
import logging
from llm_client import LLMClient

# Configure root logger to write to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_client.log'),
        logging.StreamHandler()
    ]
)

# Use client normally
client = LLMClient()
```

## Integration with Application Logging

If your application already has logging configured:

```python
import logging

# Configure your application logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# llm_client will use your configuration
from llm_client import LLMClient

client = LLMClient()
```

## Troubleshooting

### Not Seeing Logs?

1. Check if logging is configured:
   ```python
   setup_logging(level="DEBUG")
   ```

2. Check environment variable:
   ```bash
   echo $LLM_CLIENT_LOG_LEVEL
   ```

3. Ensure you're looking at stdout (not stderr)

### Too Many Logs?

1. Increase log level:
   ```python
   setup_logging(level="WARNING")
   ```

2. Or disable logging:
   ```python
   from llm_client import disable_logging
   disable_logging()
   ```

### Logs Not Formatted Correctly?

Use `force=True` to reconfigure:
```python
setup_logging(level="INFO", format_string="%(message)s", force=True)
```

## Best Practices

1. **Development**: Use `DEBUG` or `INFO` level
2. **Production**: Use `WARNING` or `ERROR` level
3. **Testing**: Consider disabling logs with `disable_logging()`
4. **Debugging**: Enable `DEBUG` temporarily when investigating issues
5. **Performance**: Higher log levels (WARNING, ERROR) have minimal performance impact

## Log Messages Reference

### Common Log Messages

| Level | Message Pattern | Meaning |
|-------|----------------|---------|
| DEBUG | `Initializing LLMClient with api_choice=...` | Client initialization started |
| DEBUG | `Found API keys for: ...` | Which API keys are available |
| INFO  | `Auto-selected API: ...` | Which provider was automatically chosen |
| INFO  | `OpenAI client initialized with model ...` | Provider successfully initialized |
| DEBUG | `Calling OpenAI API: model=..., messages=...` | Making an API call |
| DEBUG | `OpenAI response received: ... characters` | API call completed successfully |
| INFO  | `Switching provider from ... to ...` | Provider switch initiated |
| WARNING | `Tool calling in Ollama is experimental` | Feature limitation warning |
| ERROR | `OpenAI API key not found` | Missing required API key |
| ERROR | `Invalid provider: ...` | Invalid provider name specified |

## Migration from Print Statements

All `print()` statements have been replaced with appropriate logging calls:

- `print()` for info → `logger.info()`
- `print()` for debugging → `logger.debug()`
- `print()` for errors → `logger.error()`
- `print()` for warnings → `logger.warning()`

This provides much better control and flexibility for managing output.
