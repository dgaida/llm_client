# Architecture

This section describes the internal architecture and data flow of the LLM Client.

## System Overview

The LLM Client uses a **Strategy Pattern** to support multiple LLM providers through a unified interface. A **Factory Pattern** is used to instantiate the appropriate provider based on configuration or environment.

```mermaid
graph TD
    subgraph Client
        LLM[LLMClient]
    end

    subgraph Factory
        PF[ProviderFactory]
    end

    subgraph Providers
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

## Data Flow

When a user sends a message, it flows through the following components:

```mermaid
sequenceDiagram
    participant User
    participant Client as LLMClient
    participant TC as TokenCounter
    participant P as Provider
    participant API as External API

    User->>Client: chat_completion(messages)
    Client->>TC: count_tokens(messages)
    TC-->>Client: token_count
    Client->>P: _chat_completion_impl(messages)
    P->>API: POST /chat/completions
    API-->>P: JSON Response
    P-->>Client: parsed_text
    Client-->>User: result_string
```

## Provider Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Uninitialized
    Uninitialized --> Initializing: __init__()
    Initializing --> AutoDetecting: api_choice=None
    Initializing --> SpecificProvider: api_choice set
    AutoDetecting --> Ready: Provider found
    SpecificProvider --> Ready: Provider initialized
    Ready --> Processing: chat_completion()
    Processing --> Ready: Success
    Processing --> Error: Failure
    Error --> Processing: Retry (up to 3x)
    Error --> Ready: Max retries exceeded
    Ready --> [*]
```
