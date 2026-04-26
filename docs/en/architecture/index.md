# Architecture

The LLM Client is built modularly and uses proven design patterns.

## System Overview

```mermaid
graph TD
    User[User Code] --> Client[LLMClient]
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

## Data Flow

1. The user sends messages to the `LLMClient`.  
2. The `LLMClient` uses the `ProviderFactory` to instantiate the correct provider.  
3. The provider prepares the request for the specific API.  
4. The response is normalized and returned to the user.  
