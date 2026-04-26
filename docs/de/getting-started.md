# Erste Schritte

In diesem Guide erfährst du, wie du den LLM Client schnell in Betrieb nimmst.

## 1. Installation

Der erste Schritt ist die Installation des LLM Clients. Dies kann einfach über pip erfolgen:

```bash
pip install llm-client
```

Weitere Details und optionale Abhängigkeiten findest du im [Installations-Leitfaden](installation.md).

## 2. API-Keys konfigurieren

Um die meisten LLM-Provider nutzen zu können, benötigen Sie einen API-Key. Erstellen Sie eine `secrets.env`-Datei in Ihrem Projektverzeichnis:

```bash
# OpenAI
OPENAI_API_KEY=sk-xxxxxxxx

# Groq (optional)
GROQ_API_KEY=gsk-xxxxxxxx

# Google Gemini (optional)
GEMINI_API_KEY=AIzaSy-xxxxxxxx

# ODER: Generischer Key (automatische Erkennung)
API_KEY=sk-xxxxxxxx
```

Detaillierte Anweisungen zum Erhalt von Keys für die einzelnen Provider finden Sie im [Provider-Abschnitt](usage/providers/index.md).

## 3. Erstes Beispiel

```python
from llm_client import LLMClient

# Automatische API-Erkennung
client = LLMClient()

messages = [
    {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
    {"role": "user", "content": "Was ist Machine Learning?"}
]

response = client.chat_completion(messages)
print(response)
```

## Kernkonzepte

### Automatische Provider-Auswahl

Der Client wählt automatisch den ersten verfügbaren API-Key aus:

1. **OpenAI** (wenn `OPENAI_API_KEY` gesetzt ist)  
2. **Groq** (wenn `GROQ_API_KEY` gesetzt ist)  
3. **Gemini** (wenn `GEMINI_API_KEY` gesetzt ist)  
4. **Automatische Erkennung** (wenn `API_KEY` gesetzt ist - erkennt OpenAI, Groq, Gemini anhand des Präfixes)  
5. **Ollama** (Fallback, benötigt lokale Installation)  

### Manuelle Auswahl

```python
# Bestimmten Provider erzwingen
client = LLMClient(api_choice="gemini")

# Mit benutzerdefiniertem Modell und Parametern
client = LLMClient(
    api_choice="openai",
    llm="gpt-4o",
    temperature=0.5,
    max_tokens=2048
)
```

## Wichtige Funktionen

- **Chat Completion**: Standard-Anfragen an LLMs.  
- **Streaming**: Erhalte Antworten in Echtzeit.  
- **Token-Zählung**: Behalte die Kosten im Blick.  
- **Provider-Wechsel**: Wechsle die API zur Laufzeit.  
- **Konfigurationsdateien**: Lade Einstellungen aus YAML oder JSON.  
- **Async-Support**: Nutze `async/await` für performante Anwendungen.  

## Nächste Schritte

- [Features](features.md) - Übersicht der Funktionen  
- [API-Referenz](api/index.md) - Ausführliche Dokumentation der Klassen  
- [Beispiele](usage/basic-usage.md) - Praxisnahe Anwendungsfälle  
