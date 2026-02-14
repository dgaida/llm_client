# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---
<!-- commitizen-marker -->

## v0.4.4 (2026-02-14)

### Fix

- move assets to docs/ to fix broken image links in documentation

## v0.4.3 (2026-02-13)

## v0.4.2 (2026-02-13)

### Feat

- implement professional bilingual documentation ecosystem
- implement professional bilingual documentation ecosystem

## v0.4.1 (2026-02-12)

## v0.4.0 (2026-02-11)

### Feat

- add automatic versioning and changelog updates

### Fix

- update branch references from main to master
- enable emoji and icon support in documentation
- resolve async provider test failures and CI issues
- resolve async provider test failures and CI issues
- **tests**: resolve failing async provider tests
- **tests**: make test_detect_file_type_unsupported deterministic

---

## [0.3.0] - 2025-01-XX

### Added

#### Token Counting
- ✨ **Token counting with tiktoken** - Accurate token counting for all messages
- 📊 `count_tokens()` method for counting tokens in message lists
- 📊 `count_string_tokens()` method for counting tokens in plain text
- 🔄 Automatic fallback to estimation when tiktoken is not available
- 📦 Support for all GPT models (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)

#### Async Support
- ⚡ **Full async/await support** for chat completions
- 🔄 `achat_completion()` - Async chat completion method
- 🔄 `achat_completion_stream()` - Async streaming support
- 🔄 `achat_completion_with_tools()` - Async tool calling support
- 📦 Async providers: `AsyncOpenAIProvider`, `AsyncGroqProvider`, `AsyncGeminiProvider`
- 🎯 `use_async=True` parameter for creating async clients

#### Configuration Files
- 📁 **YAML/JSON configuration file support**
- 🔧 `LLMConfig` class for managing configurations
- 🔧 `from_config()` class method to load client from config files
- 🔧 `generate_config_template()` utility function
- 🔧 `create_default_config()` helper function
- ✅ Configuration validation with detailed error messages
- 🔄 Support for multiple provider configurations in one file
- 🌍 Global settings with per-provider overrides

#### Ollama Cloud Support
- ☁️ **Ollama Cloud API integration** - Access to cloud-hosted Ollama models
- 🔄 **Automatic cloud detection** from model names ending with `-cloud`
- 🔑 Support for `OLLAMA_API_KEY` environment variable
- 🎯 `use_ollama_cloud=True` parameter for explicit cloud mode
- 🌐 `ollama_host` parameter for custom Ollama endpoints
- 🔀 Seamless switching between local and cloud Ollama instances
- 📝 Example: `ollama_cloud_examples.py` demonstrating all cloud features

### Changed
- 📚 Enhanced documentation with examples for all new features
- 🧪 Expanded test suite with >92% coverage
- 📦 Updated dependencies: `tiktoken`, `pyyaml`, `asyncio`
- 🔧 Improved type hints throughout the codebase

### Examples
- 📖 `examples/usage_examples.py` - Demonstrates token counting, async, and config features
- 📖 `examples/ollama_cloud_examples.py` - Comprehensive Ollama Cloud usage examples
- 📖 Updated all existing examples with new capabilities

### Dependencies
- ➕ Added `tiktoken` for accurate token counting
- ➕ Added `pyyaml` for YAML configuration support
- ➕ Added `asyncio` for async support

### Maintained
- ✅ 100% backward compatibility
- ✅ All existing functionality preserved
- ✅ No breaking changes

---

## [0.2.0] - 2024-12-XX

### Added

#### Response Streaming
- ✨ **Response streaming support** for all providers (OpenAI, Groq, Gemini, Ollama)
- 🔄 `chat_completion_stream()` method for real-time token streaming
- 📦 Generator-based API for memory-efficient streaming
- ⚡ Enables progressive response display in UIs

#### Retry Logic
- 🔄 **Automatic retry with exponential backoff**
- 🎯 Up to 3 retry attempts on transient failures
- ⏱️ Exponential backoff: 4s, 8s, 10s delays
- 📦 Powered by `tenacity` library
- 🛡️ Transparent handling of temporary API errors

#### Custom Exceptions
- 🚨 **Comprehensive exception hierarchy**
- `LLMClientError` - Base exception for all package errors
- `APIKeyNotFoundError` - Missing API key errors with context
- `ProviderNotAvailableError` - Package installation errors
- `InvalidProviderError` - Invalid provider name errors
- `ChatCompletionError` - API call failures with original error
- `StreamingNotSupportedError` - Streaming not available errors
- 📋 Detailed error messages with actionable information

#### Architecture Improvements
- 🏗️ **Strategy Pattern** implementation with provider classes
- 🏭 **Factory Pattern** for provider creation
- 🎯 `BaseProvider` abstract class for consistent interface
- 📦 Concrete providers: `OpenAIProvider`, `GroqProvider`, `GeminiProvider`, `OllamaProvider`
- 🔧 `ProviderFactory` for centralized provider management

### Changed
- 🔄 `chat_completion()` now includes automatic retry logic by default
- 🔄 Error messages are more descriptive with custom exceptions
- 🔄 Better type hints and documentation throughout
- 📚 Refactored codebase for better maintainability

### Dependencies
- ➕ Added `tenacity>=8.2.0` for retry logic

### Examples
- 📖 `examples/streaming_example.py` - Comprehensive streaming examples
- 📖 Demonstrates retry behavior and exception handling

### Tests
- 🧪 Full test coverage for streaming functionality
- 🧪 Tests for retry logic with transient failures
- 🧪 Exception handling tests
- 🧪 Provider switching with streaming support

### Maintained
- ✅ 100% backward compatibility
- ✅ All existing functionality preserved
- ✅ No breaking changes

---

## [0.1.0] - 2024-11-XX

### Added
- 🎉 Initial release
- 🤖 Support for multiple LLM providers:
  - OpenAI (GPT-4, GPT-4o, GPT-3.5-turbo)
  - Groq (Llama, Mixtral, Gemma models)
  - Google Gemini (Gemini 2.0 Flash, Gemini 2.5 Flash)
  - Ollama (local models)
- 🔍 **Automatic API detection** based on available API keys
- 🔄 **Dynamic provider switching** at runtime with `switch_provider()`
- ⚙️ **Unified interface** - one method for all LLM backends
- 🔧 **Flexible configuration** - model, temperature, max_tokens customizable
- 🔐 **Google Colab support** - automatic secret loading from userdata
- 📦 **Zero-config** - works out-of-the-box with Ollama (no API keys needed)
- 🧩 **llama-index integration** via `LLMClientAdapter`
- 🧪 **Comprehensive test suite** with >90% coverage
- 📚 **Detailed documentation** with examples and tutorials
- 📓 **Jupyter notebooks** for RAG applications

### Project Structure
- Clean architecture with provider abstraction
- CI/CD pipelines (GitHub Actions)
- Automated testing (pytest)
- Code quality tools (black, ruff, mypy, bandit)
- Pre-commit hooks for code quality

### Documentation
- Comprehensive README with examples
- Contribution guidelines (CONTRIBUTING.md)
- Test documentation (tests/README.md)
- Notebook tutorials for RAG applications

---

## Release History

- **0.3.0** - Token counting, async support, configuration files, Ollama Cloud
- **0.2.0** - Streaming, retry logic, custom exceptions
- **0.1.0** - Initial release with multi-provider support

---

## Upgrading

### From 0.2.0 to 0.3.0

No breaking changes. New features are opt-in:

```python
# Token counting (optional)
token_count = client.count_tokens(messages)

# Async support (optional)
async_client = LLMClient(use_async=True)
response = await async_client.achat_completion(messages)

# Config files (optional)
client = LLMClient.from_config("config.yaml")

# Ollama Cloud (optional)
cloud_client = LLMClient(llm="gpt-oss:120b-cloud", use_ollama_cloud=True)
```

### From 0.1.0 to 0.2.0

No breaking changes. New features work automatically:

- Retry logic is enabled by default
- Streaming available via `chat_completion_stream()`
- Custom exceptions provide better error messages
- All existing code continues to work unchanged

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
