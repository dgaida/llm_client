# Architektur

In diesem Abschnitt werden die interne Architektur und der Datenfluss des LLM Client beschrieben.

## Systemübersicht

Der LLM Client verwendet ein **Strategy Pattern**, um verschiedene LLM-Provider über ein einheitliches Interface zu unterstützen. Ein **Factory Pattern** wird verwendet, um den entsprechenden Provider basierend auf der Konfiguration oder Umgebung zu instanziieren.

```mermaid
graph TD
    subgraph Client
        LLM[LLMClient]
    end

    subgraph Factory
        PF[ProviderFactory]
    end

    subgraph Provider
        OP[OpenAIProvider]
        GP[GroqProvider]
        GE[GeminiProvider]
        OL[OllamaProvider]
    end

    LLM --> PF
    PF --> OP
    PF --> GP
    PF --> GE
    PF --> OL
```

## Datenfluss

Wenn ein Benutzer eine Nachricht sendet, durchläuft diese die folgenden Komponenten:

```mermaid
sequenceDiagram
    participant User
    participant Client as LLMClient
    participant TC as TokenCounter
    participant P as Provider
    participant API as Externe API

    User->>Client: chat_completion(messages)
    Client->>TC: count_tokens(messages)
    TC-->>Client: token_count
    Client->>P: _chat_completion_impl(messages)
    P->>API: POST /chat/completions
    API-->>P: JSON-Antwort
    P-->>Client: parsed_text
    Client-->>User: result_string
```

## Provider-Lebenszyklus

```mermaid
stateDiagram-v2
    [*] --> Uninitialisiert
    Uninitialisiert --> Initialisierung: __init__()
    Initialisierung --> AutoErkennung: api_choice=None
    Initialisierung --> SpezifischerProvider: api_choice gesetzt
    AutoErkennung --> Bereit: Provider gefunden
    SpezifischerProvider --> Bereit: Provider initialisiert
    Bereit --> Verarbeitung: chat_completion()
    Verarbeitung --> Bereit: Erfolg
    Verarbeitung --> Fehler: Fehlgeschlagen
    Fehler --> Verarbeitung: Retry (bis zu 3x)
    Fehler --> Bereit: Max Retries überschritten
    Bereit --> [*]
```
