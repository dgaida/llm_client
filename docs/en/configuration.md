# Configuration

Manage multiple provider configurations easily via YAML or JSON.

## Create Config File

```python
from llm_client.config import generate_config_template

# Generate template
generate_config_template("llm_config.yaml", format="yaml")
```

## Example Configuration

```yaml
# llm_config.yaml
default_provider: openai

global_settings:
  temperature: 0.7
  max_tokens: 512

providers:
  openai:
    model: gpt-4o-mini
    temperature: 0.7
    max_tokens: 512

  groq:
    model: llama-3.3-70b-versatile
    temperature: 0.5
    max_tokens: 1024
```

## Loading from Config

```python
from llm_client import LLMClient

# Load default provider
client = LLMClient.from_config("llm_config.yaml")
```

## Further Information

- [API Reference: LLMConfig](api/config.md)
