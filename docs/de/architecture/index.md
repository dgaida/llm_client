# Architektur

Der LLM Client ist modular aufgebaut und nutzt bewährte Entwurfsmuster.

## Systemübersicht

```mermaid
graph TD
    User[Benutzer Code] --> Client[LLMClient]
    Client --> Factory[ProviderFactory]
    Factory --> P1[OpenAIProvider]
    Factory --> P2[GroqProvider]
    Factory --> P3[GeminiProvider]
    Factory --> P4[OllamaProvider]

    subgraph "Utilities"
        Client --> Token[TokenCounter]
        Client --> Config[LLMConfig]
    end
```

## Datenfluss

1. Der Benutzer sendet Nachrichten an den `LLMClient`.
2. Der `LLMClient` nutzt die `ProviderFactory`, um den richtigen Provider zu instanziieren.
3. Der Provider bereitet die Anfrage für die spezifische API vor.
4. Die Antwort wird normalisiert und an den Benutzer zurückgegeben.
