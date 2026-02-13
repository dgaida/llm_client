# Token-Zählung

Die präzise Zählung von Tokens ist wichtig für das Kostenmanagement und die Einhaltung von Kontext-Limits.

## Verwendung

```python
from llm_client import LLMClient

client = LLMClient()

messages = [
    {"role": "user", "content": "Hallo, wie geht es dir?"}
]

# Tokens für Nachrichtenliste zählen
count = client.count_tokens(messages)
print(f"Tokens: {count}")

# Tokens für einen einfachen String zählen
string_count = client.count_string_tokens("Ein Beispieltext.")
print(f"String Tokens: {string_count}")
```

## Funktionsweise

Der LLM Client verwendet `tiktoken` für OpenAI-Modelle, was sehr präzise ist. Für andere Provider wird eine Schätzung vorgenommen, falls `tiktoken` nicht verfügbar ist oder das Modell nicht unterstützt wird.
