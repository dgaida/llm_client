with open('docs/en/features.md', 'r', encoding='utf-8') as f:
    content = f.read()

redundant_part = """LLM Client automatically detects which LLM provider to use based on available API keys:

```python
from llm_client import LLMClient

# Automatically selects first available provider:
# 1. OpenAI (if OPENAI_API_KEY set)
# 2. Groq (if GROQ_API_KEY set)
# 3. Gemini (if GEMINI_API_KEY set)
# 4. Ollama (local fallback, no key needed)
client = LLMClient()

print(f"Using: {client.api_choice}")  # e.g., "openai"
```

[:octicons-arrow-right-24: Learn more](getting-started.md#automatic-provider-selection)"""

content = content.replace(redundant_part, "")

with open('docs/en/features.md', 'w', encoding='utf-8') as f:
    f.write(content)
