# 🧪 Testing

```bash
# All tests
pytest

# With coverage report
pytest --cov=llm_client --cov-report=html

# Specific tests
pytest tests/test_llm_client.py -v

# Tests for new features
pytest tests/test_new_features.py -v
pytest tests/tests_new_features_complete.py -v
```

## Test Structure

```
tests/
├── test_llm_client.py              # Comprehensive tests for all providers
├── test_switch_provider.py         # Dynamic provider switching tests
├── test_adapter.py                 # llama-index integration tests
├── test_base_provider.py           # Abstract base class tests
├── test_providers.py               # Provider implementation tests
├── test_provider_factory.py        # Factory pattern tests
├── test_new_features.py            # Streaming and retry logic tests
├── tests_new_features_complete.py  # Token counting, async, config tests
├── test_integration.py             # End-to-end integration tests
└── README.md                       # Detailed test documentation
```

## Code Quality Checks

```bash
# Format code
black .

# Lint
ruff check .

# Auto-fix
ruff check --fix .

# Type checking
mypy llm_client
```
