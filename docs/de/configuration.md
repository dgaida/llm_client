# Konfiguration

Verwalten Sie mehrere Provider-Konfigurationen einfach via YAML oder JSON.

## Config-Datei erstellen

```python
from llm_client.config import generate_config_template

# Template generieren
generate_config_template("llm_config.yaml", format="yaml")
```

## Beispiel-Konfiguration

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

## Aus Config laden

```python
from llm_client import LLMClient

# Standard-Provider laden
client = LLMClient.from_config("llm_config.yaml")
```

## Weitere Informationen

- [API-Referenz: LLMConfig](api/config.md)  
