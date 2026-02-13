# Installation

Hier erfährst du, wie du den LLM Client installierst und für die Entwicklung einrichtest.

## Schnellinstallation

Du kannst das Paket direkt von GitHub mit pip installieren:

```bash
pip install git+https://github.com/dgaida/llm_client.git
```

## Entwicklungsinstallation

Für die lokale Entwicklung klone das Repository und installiere es im Editable-Modus:

```bash
git clone https://github.com/dgaida/llm_client.git
cd llm_client
pip install -e ".[dev]"
```

## Optionale Abhängigkeiten

Der LLM Client bietet verschiedene Extras für zusätzliche Funktionen:

```bash
# Mit LlamaIndex-Unterstützung
pip install -e ".[llama-index]"

# Mit allen Features und Entwicklungs-Tools
pip install -e ".[all]"
```

## Systemvoraussetzungen

- **Python**: 3.10 oder höher
- **Betriebssystem**: Windows, macOS oder Linux
- **Internetverbindung**: Erforderlich für Cloud-Provider (OpenAI, Groq, Gemini)
- **Ollama**: Erforderlich für die lokale Modellnutzung
