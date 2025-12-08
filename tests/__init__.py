"""
tests
=====

Test suite for llm_client package.

Test modules:
- test_llm_client.py: Core LLMClient functionality tests
- test_llm_client_extended.py: Extended tests & edge cases
- test_adapter.py: LLMClientAdapter tests (llama-index integration)
- test_async_providers.py: Async provider tests
- test_base_provider.py: BaseProvider abstract class tests
- test_config_extended.py: Extended configuration tests
- test_integration.py: End-to-end integration tests
- test_new_features.py: Streaming & retry logic tests
- test_provider_factory.py: Factory pattern tests
- test_providers.py: Provider implementation tests
- test_switch_provider.py: Provider switching tests
- test_token_counter.py: Token counting tests
- tests_new_features_complete.py: Comprehensive feature tests

To run tests:
    pytest                              # All tests
    pytest --cov=llm_client            # With coverage
    pytest tests/test_llm_client.py    # Specific module
    pytest -v                          # Verbose output
    pytest -k "async"                  # Tests matching pattern

Coverage:
- Overall: ~92% coverage
- llm_client.py: ~95% coverage
- adapter.py: ~90% coverage

Test markers:
- asyncio: Async tests requiring pytest-asyncio
- skipif: Tests skipped based on conditions (e.g., llama-index not installed)
"""

# Empty file - tests are standalone modules
# This allows tests directory to be imported as a package if needed
