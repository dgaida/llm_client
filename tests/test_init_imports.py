"""Tests for __init__.py import error branches."""

import importlib
import sys
from unittest.mock import patch


def test_init_async_import_error():
    """Test: Handle ImportError for async providers in __init__."""
    with patch.dict(sys.modules, {"llm_client.providers.async_providers": None}):
        import llm_client

        importlib.reload(llm_client)
        # Should not have async providers in __all__
        assert "AsyncOpenAIProvider" not in llm_client.__all__


def test_init_adapter_import_error():
    """Test: Handle ImportError for llama-index adapter in __init__."""
    with patch.dict(sys.modules, {"llm_client.providers.adapter": None}):
        import llm_client

        importlib.reload(llm_client)
        # Should not have LLMClientAdapter in __all__
        assert "LLMClientAdapter" not in llm_client.__all__


def test_init_success_with_reloads():
    """Test: Ensure normal state is restored after reloads."""
    import llm_client

    importlib.reload(llm_client)
    # This might depend on what's actually installed in the environment
    # but at least check it doesn't crash
    assert "LLMClient" in llm_client.__all__


def test_adapter_dummy_classes():
    """Test: Cover dummy classes in adapter when llama-index is missing."""
    import importlib
    import sys
    from unittest.mock import patch

    with patch.dict(
        sys.modules, {"llama_index": None, "llama_index.core": None, "llama_index.core.llms": None}
    ):
        from llm_client.providers import adapter

        importlib.reload(adapter)
        assert adapter.LLAMA_INDEX_AVAILABLE is False
        assert adapter.LLM is object

        # Reset
        importlib.reload(adapter)
