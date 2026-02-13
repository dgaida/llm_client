# Provider Switching Examples

Dynamic provider switching is one of the most powerful features of LLM Client. This guide demonstrates various strategies and use cases.

## Table of Contents

- [Basic Switching](#basic-switching)
- [Fallback Strategies](#fallback-strategies)
- [Cost Optimization](#cost-optimization)
- [Performance Optimization](#performance-optimization)
- [Quality vs Speed Trade-offs](#quality-vs-speed-trade-offs)
- [Multi-Provider Workflows](#multi-provider-workflows)
- [A/B Testing](#ab-testing)
- [Load Balancing](#load-balancing)

## Basic Switching

### Simple Provider Switch

```python
from llm_client import LLMClient

# Start with OpenAI
client = LLMClient(api_choice="openai", llm="gpt-4o-mini")
print(f"Using: {client.api_choice} - {client.llm}")

messages = [{"role": "user", "content": "Hello!"}]
response1 = client.chat_completion(messages)

# Switch to Groq
client.switch_provider("groq", llm="llama-3.3-70b-versatile")
print(f"Using: {client.api_choice} - {client.llm}")

response2 = client.chat_completion(messages)

# Switch to Gemini
client.switch_provider("gemini", llm="gemini-2.5-flash")
response3 = client.chat_completion(messages)
```

### Switch with Parameter Updates

```python
# Change provider and parameters simultaneously
client = LLMClient(api_choice="openai")

# Switch to Groq with custom temperature
client.switch_provider(
    "groq",
    llm="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1024
)

# Switch to Gemini with high creativity
client.switch_provider(
    "gemini",
    llm="gemini-2.5-pro",
    temperature=1.5,
    max_tokens=2048
)
```

## Fallback Strategies

### Simple Fallback

```python
from llm_client import LLMClient
from llm_client.exceptions import ChatCompletionError

messages = [{"role": "user", "content": "Explain quantum computing"}]

# Try primary provider
client = LLMClient(api_choice="openai")

try:
    response = client.chat_completion(messages)
    print("✅ OpenAI succeeded")
except ChatCompletionError as e:
    print(f"❌ OpenAI failed: {e}")

    # Fallback to secondary provider
    client.switch_provider("groq")
    response = client.chat_completion(messages)
    print("✅ Groq succeeded")

print(f"Response: {response}")
```

### Multi-Level Fallback

```python
from llm_client import LLMClient
from llm_client.exceptions import ChatCompletionError, APIKeyNotFoundError

def robust_completion(messages, providers=None):
    """Try multiple providers with fallback chain."""

    if providers is None:
        providers = [
            ("openai", "gpt-4o-mini"),
            ("groq", "llama-3.3-70b-versatile"),
            ("gemini", "gemini-2.5-flash"),
            ("ollama", "llama3.2:3b")
        ]

    client = LLMClient()

    for api_choice, model in providers:
        try:
            print(f"Trying {api_choice}...")
            client.switch_provider(api_choice, llm=model)
            response = client.chat_completion(messages)
            print(f"✅ Success with {api_choice}")
            return response

        except (ChatCompletionError, APIKeyNotFoundError) as e:
            print(f"❌ {api_choice} failed: {type(e).__name__}")
            continue

    raise RuntimeError("All providers failed")

# Usage
messages = [{"role": "user", "content": "What is machine learning?"}]
response = robust_completion(messages)
```

### Provider Health Check

```python
from llm_client import LLMClient
from llm_client.exceptions import ChatCompletionError

def check_provider_health(api_choice, model=None):
    """Test if a provider is responding."""
    client = LLMClient(api_choice=api_choice, llm=model)
    test_message = [{"role": "user", "content": "Hi"}]

    try:
        client.chat_completion(test_message)
        return True
    except ChatCompletionError:
        return False

# Check all providers
providers = [
    ("openai", "gpt-4o-mini"),
    ("groq", "llama-3.3-70b-versatile"),
    ("gemini", "gemini-2.5-flash"),
]

healthy_providers = []
for api, model in providers:
    if check_provider_health(api, model):
        healthy_providers.append((api, model))
        print(f"✅ {api} is healthy")
    else:
        print(f"❌ {api} is down")

# Use first healthy provider
if healthy_providers:
    api, model = healthy_providers[0]
    client = LLMClient(api_choice=api, llm=model)
```

## Cost Optimization

### Dynamic Cost-Based Selection

```python
from llm_client import LLMClient, TokenCounter

# Approximate costs (per 1M tokens)
COSTS = {
    ("openai", "gpt-4o"): (2.50, 10.00),
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("groq", "llama-3.3-70b-versatile"): (0.59, 0.79),
    ("gemini", "gemini-2.5-pro"): (1.25, 5.00),
    ("gemini", "gemini-2.5-flash"): (0.075, 0.30),
    ("ollama", "llama3.1:8b"): (0.0, 0.0),  # Free!
}

def estimate_cost(provider, model, input_tokens, output_tokens):
    """Estimate cost for a completion."""
    input_cost, output_cost = COSTS.get((provider, model), (0, 0))
    total = (input_tokens / 1_000_000) * input_cost + \
            (output_tokens / 1_000_000) * output_cost
    return total

def choose_cost_effective_provider(messages, budget=0.01):
    """Choose cheapest provider within budget."""

    token_count = TokenCounter.count_tokens(messages)
    estimated_output = 200  # Estimate response length

    # Check each provider's cost
    for (provider, model), (in_cost, out_cost) in COSTS.items():
        cost = estimate_cost(provider, model, token_count, estimated_output)

        if cost <= budget:
            print(f"Selected {provider}/{model} (estimated ${cost:.4f})")
            return provider, model

    # If nothing fits budget, use free Ollama
    print("Using free Ollama to stay within budget")
    return "ollama", "llama3.1:8b"

# Usage
messages = [
    {"role": "user", "content": "Write a long essay about AI"}
]

provider, model = choose_cost_effective_provider(messages, budget=0.005)
client = LLMClient(api_choice=provider, llm=model)
response = client.chat_completion(messages)
```

### Tiered Pricing Strategy

```python
from llm_client import LLMClient

class TieredLLMClient:
    """Client that selects provider based on task complexity."""

    def __init__(self):
        self.client = LLMClient()

    def completion(self, messages, complexity="medium"):
        """
        Execute completion with provider selection based on complexity.

        Args:
            messages: Chat messages
            complexity: "simple", "medium", or "complex"
        """

        if complexity == "simple":
            # Use cheapest/fastest option
            self.client.switch_provider("groq", llm="llama-3.3-70b-versatile")
            print("💰 Using Groq (cost-effective)")

        elif complexity == "medium":
            # Balanced option
            self.client.switch_provider("gemini", llm="gemini-2.5-flash")
            print("⚖️ Using Gemini Flash (balanced)")

        else:  # complex
            # Most capable option
            self.client.switch_provider("openai", llm="gpt-4o")
            print("🚀 Using GPT-4o (high quality)")

        return self.client.chat_completion(messages)

# Usage
client = TieredLLMClient()

# Simple question
simple = [{"role": "user", "content": "What is 2+2?"}]
client.completion(simple, complexity="simple")

# Complex analysis
complex_task = [{"role": "user", "content": "Analyze the geopolitical..."}]
client.completion(complex_task, complexity="complex")
```

## Performance Optimization

### Speed-First Selection

```python
from llm_client import LLMClient
import time

def benchmark_provider(api_choice, model, messages):
    """Measure response time for a provider."""
    client = LLMClient(api_choice=api_choice, llm=model)

    start = time.time()
    response = client.chat_completion(messages)
    elapsed = time.time() - start

    return elapsed, response

# Test message
test_messages = [{"role": "user", "content": "Count from 1 to 5"}]

# Benchmark providers
providers = [
    ("groq", "llama-3.3-70b-versatile"),
    ("openai", "gpt-4o-mini"),
    ("gemini", "gemini-2.5-flash"),
]

results = []
for api, model in providers:
    try:
        elapsed, response = benchmark_provider(api, model, test_messages)
        results.append((elapsed, api, model))
        print(f"{api:10s} - {model:30s}: {elapsed:.2f}s")
    except Exception as e:
        print(f"{api:10s} - Failed: {e}")

# Use fastest provider
if results:
    elapsed, fastest_api, fastest_model = min(results)
    print(f"\n🏆 Fastest: {fastest_api}/{fastest_model} ({elapsed:.2f}s)")

    client = LLMClient(api_choice=fastest_api, llm=fastest_model)
```

### Streaming for Perceived Performance

```python
from llm_client import LLMClient

def smart_streaming_client(messages, use_streaming=True):
    """Use streaming for better UX on long responses."""

    # Estimate response length from query
    query_length = len(messages[-1]["content"])

    # Use streaming for longer queries (likely longer responses)
    if query_length > 100 and use_streaming:
        client = LLMClient(api_choice="groq")  # Fast streaming
        print("Using streaming for better UX...")

        for chunk in client.chat_completion_stream(messages):
            print(chunk, end="", flush=True)
        print()
    else:
        # Short query, non-streaming is fine
        client = LLMClient(api_choice="groq")
        response = client.chat_completion(messages)
        print(response)

# Usage
short_query = [{"role": "user", "content": "What is AI?"}]
long_query = [{"role": "user", "content": "Write a detailed explanation of..."}]

smart_streaming_client(short_query, use_streaming=False)
smart_streaming_client(long_query, use_streaming=True)
```

## Quality vs Speed Trade-offs

### Adaptive Quality Selection

```python
from llm_client import LLMClient

class AdaptiveClient:
    """Automatically select provider based on task requirements."""

    def __init__(self):
        self.client = LLMClient()

    def completion(self, messages, priority="balanced"):
        """
        Args:
            priority: "speed", "balanced", or "quality"
        """

        if priority == "speed":
            # Fastest provider
            self.client.switch_provider(
                "groq",
                llm="llama-3.3-70b-versatile",
                temperature=0.5
            )
            print("⚡ Speed priority: Groq")

        elif priority == "quality":
            # Highest quality
            self.client.switch_provider(
                "openai",
                llm="gpt-4o",
                temperature=0.7
            )
            print("💎 Quality priority: GPT-4o")

        else:  # balanced
            # Good balance
            self.client.switch_provider(
                "gemini",
                llm="gemini-2.5-flash",
                temperature=0.7
            )
            print("⚖️ Balanced: Gemini Flash")

        return self.client.chat_completion(messages)

# Usage
client = AdaptiveClient()

# Real-time chat needs speed
chat_msg = [{"role": "user", "content": "Quick question..."}]
client.completion(chat_msg, priority="speed")

# Important document needs quality
doc_msg = [{"role": "user", "content": "Write business proposal..."}]
client.completion(doc_msg, priority="quality")
```

## Multi-Provider Workflows

### Parallel Provider Comparison

```python
from llm_client import LLMClient
from concurrent.futures import ThreadPoolExecutor

def compare_providers(messages):
    """Get responses from multiple providers in parallel."""

    providers = [
        ("openai", "gpt-4o-mini"),
        ("groq", "llama-3.3-70b-versatile"),
        ("gemini", "gemini-2.5-flash"),
    ]

    def get_response(provider_info):
        api, model = provider_info
        try:
            client = LLMClient(api_choice=api, llm=model)
            response = client.chat_completion(messages)
            return (api, response)
        except Exception as e:
            return (api, f"Error: {e}")

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(get_response, providers))

    return dict(results)

# Usage
messages = [{"role": "user", "content": "Explain blockchain"}]
responses = compare_providers(messages)

for provider, response in responses.items():
    print(f"\n{provider.upper()}:")
    print("-" * 50)
    print(response[:200] + "..." if len(response) > 200 else response)
```

### Chain of Thought with Multiple Providers

```python
from llm_client import LLMClient

def multi_provider_chain(question):
    """Use different providers for different stages."""

    client = LLMClient()

    # Stage 1: Fast provider for initial thoughts
    print("Stage 1: Generating initial thoughts (Groq)...")
    client.switch_provider("groq", llm="llama-3.3-70b-versatile")

    initial = client.chat_completion([
        {"role": "user", "content": f"List key points about: {question}"}
    ])
    print(f"Initial thoughts:\n{initial}\n")

    # Stage 2: Quality provider for detailed analysis
    print("Stage 2: Detailed analysis (GPT-4o)...")
    client.switch_provider("openai", llm="gpt-4o")

    detailed = client.chat_completion([
        {
            "role": "system",
            "content": f"Based on these points:\n{initial}\n\nProvide detailed analysis."
        },
        {"role": "user", "content": question}
    ])
    print(f"Detailed analysis:\n{detailed}\n")

    # Stage 3: Summarization with balanced provider
    print("Stage 3: Final summary (Gemini)...")
    client.switch_provider("gemini", llm="gemini-2.5-flash")

    summary = client.chat_completion([
        {
            "role": "user",
            "content": f"Summarize concisely:\n{detailed}"
        }
    ])
    print(f"Summary:\n{summary}")

    return summary

# Usage
multi_provider_chain("What is the future of AI?")
```

## A/B Testing

### Provider A/B Testing

```python
from llm_client import LLMClient
import random

class ABTestClient:
    """Client that randomly selects provider for A/B testing."""

    def __init__(self, providers_a, providers_b, split=0.5):
        """
        Args:
            providers_a: List of (api, model) tuples for group A
            providers_b: List of (api, model) tuples for group B
            split: Probability of selecting group A (0.0 to 1.0)
        """
        self.providers_a = providers_a
        self.providers_b = providers_b
        self.split = split
        self.client = LLMClient()
        self.results = {"A": [], "B": []}

    def completion(self, messages, user_id=None):
        """Execute completion with A/B testing."""

        # Consistent selection for same user
        if user_id:
            random.seed(hash(user_id))

        # A/B selection
        if random.random() < self.split:
            group = "A"
            api, model = random.choice(self.providers_a)
        else:
            group = "B"
            api, model = random.choice(self.providers_b)

        # Execute
        self.client.switch_provider(api, llm=model)
        response = self.client.chat_completion(messages)

        # Log for analysis
        self.results[group].append({
            "user_id": user_id,
            "provider": api,
            "model": model,
            "response_length": len(response)
        })

        return response, group

    def get_stats(self):
        """Get A/B test statistics."""
        return {
            "Group A": {
                "count": len(self.results["A"]),
                "avg_length": sum(r["response_length"] for r in self.results["A"]) / len(self.results["A"]) if self.results["A"] else 0
            },
            "Group B": {
                "count": len(self.results["B"]),
                "avg_length": sum(r["response_length"] for r in self.results["B"]) / len(self.results["B"]) if self.results["B"] else 0
            }
        }

# Usage
ab_client = ABTestClient(
    providers_a=[("openai", "gpt-4o-mini")],
    providers_b=[("groq", "llama-3.3-70b-versatile")],
    split=0.5
)

# Simulate multiple users
for user_id in range(100):
    messages = [{"role": "user", "content": f"Question {user_id}"}]
    response, group = ab_client.completion(messages, user_id=f"user_{user_id}")
    print(f"User {user_id}: Group {group}")

# Analyze results
stats = ab_client.get_stats()
print("\nA/B Test Results:")
print(stats)
```

## Load Balancing

### Simple Round-Robin Load Balancer

```python
from llm_client import LLMClient
from itertools import cycle

class LoadBalancedClient:
    """Distribute requests across multiple providers."""

    def __init__(self, providers):
        """
        Args:
            providers: List of (api, model) tuples
        """
        self.provider_cycle = cycle(providers)
        self.client = LLMClient()
        self.request_count = {}

    def completion(self, messages):
        """Execute with next provider in rotation."""

        # Get next provider
        api, model = next(self.provider_cycle)

        # Switch and execute
        self.client.switch_provider(api, llm=model)
        response = self.client.chat_completion(messages)

        # Track usage
        key = f"{api}/{model}"
        self.request_count[key] = self.request_count.get(key, 0) + 1

        print(f"Used: {key} (Total: {self.request_count[key]})")

        return response

    def get_usage_stats(self):
        """Get load distribution statistics."""
        return self.request_count

# Usage
lb_client = LoadBalancedClient([
    ("openai", "gpt-4o-mini"),
    ("groq", "llama-3.3-70b-versatile"),
    ("gemini", "gemini-2.5-flash"),
])

# Make multiple requests
for i in range(9):
    messages = [{"role": "user", "content": f"Request {i}"}]
    lb_client.completion(messages)

print("\nLoad Distribution:")
print(lb_client.get_usage_stats())
```

## Best Practices

### 1. Always Have a Fallback

```python
# Good: Multiple fallback options
providers = ["openai", "groq", "gemini", "ollama"]

# Bad: Single provider with no fallback
providers = ["openai"]
```

### 2. Monitor Provider Health

```python
# Periodically check provider availability
def health_check_loop():
    while True:
        for provider in ["openai", "groq", "gemini"]:
            if not check_provider_health(provider):
                alert_admin(f"{provider} is down!")
        time.sleep(300)  # Check every 5 minutes
```

### 3. Log Provider Usage

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def logged_completion(client, messages):
    """Completion with logging."""
    provider = client.api_choice
    model = client.llm

    logger.info(f"Starting completion: {provider}/{model}")
    start = time.time()

    try:
        response = client.chat_completion(messages)
        elapsed = time.time() - start
        logger.info(f"Success: {provider}/{model} ({elapsed:.2f}s)")
        return response
    except Exception as e:
        logger.error(f"Failed: {provider}/{model} - {e}")
        raise
```

### 4. Consider Rate Limits

```python
from time import sleep

class RateLimitedClient:
    """Client that respects rate limits."""

    def __init__(self, requests_per_minute=60):
        self.client = LLMClient()
        self.requests_per_minute = requests_per_minute
        self.last_request_time = 0

    def completion(self, messages):
        """Execute with rate limiting."""
        # Calculate required delay
        min_interval = 60.0 / self.requests_per_minute
        elapsed = time.time() - self.last_request_time

        if elapsed < min_interval:
            sleep(min_interval - elapsed)

        response = self.client.chat_completion(messages)
        self.last_request_time = time.time()

        return response
```

## Conclusion

Dynamic provider switching enables:

✅ **Resilience** - Automatic failover to working providers
✅ **Cost optimization** - Select cheapest option for each task
✅ **Performance tuning** - Balance speed vs quality
✅ **A/B testing** - Compare provider performance
✅ **Load balancing** - Distribute across multiple APIs

Choose the strategy that best fits your use case!
