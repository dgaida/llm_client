"""Unit tests for BaseProvider abstract class."""

from unittest.mock import MagicMock

import pytest

from llm_client.providers.base_provider import BaseProvider


class ConcreteProvider(BaseProvider):
    """Concrete implementation of BaseProvider for testing."""

    def _initialize_client(self, **kwargs):
        """Initialize a mock client."""
        self.client = MagicMock()
        self.init_kwargs = kwargs

    def _chat_completion_impl(self, messages):
        """Return a mock response."""
        return "Mock response"

    @staticmethod
    def get_default_model():
        """Return a default model name."""
        return "test-model"

    @staticmethod
    def is_available():
        """Return True for testing."""
        return True


class TestBaseProvider:
    """Tests for BaseProvider abstract base class."""

    def test_cannot_instantiate_base_provider_directly(self):
        """Test: Cannot instantiate BaseProvider directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseProvider(llm="test-model")

    def test_concrete_provider_initialization(self):
        """Test: Concrete provider can be instantiated."""
        provider = ConcreteProvider(
            llm="test-model", temperature=0.5, max_tokens=1024, custom_param="value"
        )

        assert provider.llm == "test-model"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 1024
        assert provider.client is not None
        assert provider.init_kwargs == {"custom_param": "value"}

    def test_default_parameters(self):
        """Test: Default parameters are set correctly."""
        provider = ConcreteProvider(llm="test-model")

        assert provider.llm == "test-model"
        assert provider.temperature == 0.7
        assert provider.max_tokens == 512

    def test_repr_method(self):
        """Test: __repr__ returns correct string representation."""
        provider = ConcreteProvider(llm="custom-model", temperature=0.3, max_tokens=2048)
        repr_str = repr(provider)

        assert "ConcreteProvider" in repr_str
        assert "custom-model" in repr_str
        assert "0.3" in repr_str

    def test_chat_completion_method(self):
        """Test: chat_completion method works."""
        provider = ConcreteProvider(llm="test-model")
        messages = [{"role": "user", "content": "Hello"}]
        response = provider.chat_completion(messages)

        assert response == "Mock response"

    def test_get_default_model_method(self):
        """Test: get_default_model static method works."""
        assert ConcreteProvider.get_default_model() == "test-model"

    def test_is_available_method(self):
        """Test: is_available static method works."""
        assert ConcreteProvider.is_available() is True

    def test_kwargs_passed_to_initialize_client(self):
        """Test: Additional kwargs are passed to _initialize_client."""
        provider = ConcreteProvider(
            llm="test-model", temperature=0.7, max_tokens=512, api_key="test-key", option1="value1"
        )

        assert provider.init_kwargs == {"api_key": "test-key", "option1": "value1"}

    def test_multiple_instances_independent(self):
        """Test: Multiple instances are independent."""
        provider1 = ConcreteProvider(llm="model-1", temperature=0.3)
        provider2 = ConcreteProvider(llm="model-2", temperature=0.8)

        assert provider1.llm == "model-1"
        assert provider2.llm == "model-2"
        assert provider1.temperature == 0.3
        assert provider2.temperature == 0.8


class IncompleteProvider(BaseProvider):
    """Provider missing required abstract methods."""

    def _initialize_client(self, **kwargs):
        """Initialize client."""
        self.client = MagicMock()


class TestAbstractMethodEnforcement:
    """Tests for abstract method enforcement."""

    def test_missing_chat_completion_impl_raises_error(self):
        """Test: Cannot instantiate without _chat_completion_impl method."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProvider(llm="test")

    def test_all_abstract_methods_must_be_implemented(self):
        """Test: All abstract methods must be implemented."""

        class AlmostCompleteProvider(BaseProvider):
            def _initialize_client(self, **kwargs):
                self.client = MagicMock()

            def _chat_completion_impl(self, messages):
                return "response"

            @staticmethod
            def get_default_model():
                return "model"

            # Missing is_available

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AlmostCompleteProvider(llm="test")


class ParameterValidationProvider(ConcreteProvider):
    """Provider with parameter validation."""

    def _initialize_client(self, **kwargs):
        """Initialize with validation."""
        if "api_key" not in kwargs:
            raise ValueError("API key required")
        super()._initialize_client(**kwargs)


class TestProviderParameterValidation:
    """Tests for parameter validation in providers."""

    def test_provider_can_validate_parameters(self):
        """Test: Provider can validate required parameters."""
        with pytest.raises(ValueError, match="API key required"):
            ParameterValidationProvider(llm="test-model")

    def test_provider_with_valid_parameters(self):
        """Test: Provider initializes with valid parameters."""
        provider = ParameterValidationProvider(llm="test-model", api_key="valid-key")
        assert provider.client is not None


class TestProviderClientAttribute:
    """Tests for client attribute handling."""

    def test_client_attribute_initialized(self):
        """Test: Client attribute is set during initialization."""
        provider = ConcreteProvider(llm="test-model")
        assert provider.client is not None
        assert isinstance(provider.client, MagicMock)

    def test_client_can_be_replaced(self):
        """Test: Client attribute can be replaced after initialization."""
        provider = ConcreteProvider(llm="test-model")
        original_client = provider.client

        new_client = MagicMock()
        provider.client = new_client

        assert provider.client is new_client
        assert provider.client is not original_client


class TestProviderTypeHints:
    """Tests for type hints and type safety."""

    def test_llm_parameter_is_string(self):
        """Test: llm parameter accepts string."""
        provider = ConcreteProvider(llm="string-model")
        assert isinstance(provider.llm, str)

    def test_temperature_parameter_is_float(self):
        """Test: temperature parameter accepts float."""
        provider = ConcreteProvider(llm="test", temperature=0.5)
        assert isinstance(provider.temperature, float)

    def test_max_tokens_parameter_is_int(self):
        """Test: max_tokens parameter accepts int."""
        provider = ConcreteProvider(llm="test", max_tokens=1024)
        assert isinstance(provider.max_tokens, int)

    def test_messages_parameter_type(self):
        """Test: chat_completion accepts list of dicts."""
        ConcreteProvider(llm="test")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        assert isinstance(messages, list)
        assert all(isinstance(m, dict) for m in messages)
        assert all("role" in m and "content" in m for m in messages)


class MockFailingProvider(ConcreteProvider):
    """Provider that simulates initialization failure."""

    def _initialize_client(self, **kwargs):
        """Simulate client initialization failure."""
        raise RuntimeError("Client initialization failed")


class TestProviderErrorHandling:
    """Tests for error handling in providers."""

    def test_initialization_error_propagates(self):
        """Test: Initialization errors propagate correctly."""
        with pytest.raises(RuntimeError, match="Client initialization failed"):
            MockFailingProvider(llm="test-model")

    def test_repr_with_special_characters(self):
        """Test: __repr__ handles special characters in model name."""
        provider = ConcreteProvider(llm="model/with/slashes", temperature=0.7)
        repr_str = repr(provider)

        assert "ConcreteProvider" in repr_str
        assert "model/with/slashes" in repr_str


class TestProviderInheritance:
    """Tests for provider inheritance patterns."""

    def test_subclass_can_override_repr(self):
        """Test: Subclass can override __repr__ method."""

        class CustomReprProvider(ConcreteProvider):
            def __repr__(self):
                return f"Custom({self.llm})"

        provider = CustomReprProvider(llm="test-model")
        assert repr(provider) == "Custom(test-model)"

    def test_subclass_inherits_base_attributes(self):
        """Test: Subclass inherits all base attributes."""

        class ExtendedProvider(ConcreteProvider):
            def __init__(self, llm, temperature=0.7, max_tokens=512, **kwargs):
                super().__init__(llm, temperature, max_tokens, **kwargs)
                self.extended_attribute = "extended"

        provider = ExtendedProvider(llm="test-model")
        assert hasattr(provider, "llm")
        assert hasattr(provider, "temperature")
        assert hasattr(provider, "max_tokens")
        assert hasattr(provider, "client")
        assert hasattr(provider, "extended_attribute")

    def test_multiple_inheritance_levels(self):
        """Test: Multiple levels of inheritance work correctly."""

        class MiddleProvider(ConcreteProvider):
            def middle_method(self):
                return "middle"

        class LeafProvider(MiddleProvider):
            def leaf_method(self):
                return "leaf"

        provider = LeafProvider(llm="test-model")
        assert provider.middle_method() == "middle"
        assert provider.leaf_method() == "leaf"
        assert provider.chat_completion([]) == "Mock response"


class TestRetryLogic:
    """Tests for retry logic in chat_completion."""

    def test_chat_completion_with_retry_on_success(self):
        """Test: chat_completion succeeds on first try."""
        provider = ConcreteProvider(llm="test-model")
        messages = [{"role": "user", "content": "Hello"}]

        response = provider.chat_completion(messages)
        assert response == "Mock response"

    def test_chat_completion_retries_on_failure(self):
        """Test: chat_completion retries on transient failures."""

        class FailingProvider(ConcreteProvider):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.attempt_count = 0

            def _chat_completion_impl(self, messages):
                self.attempt_count += 1
                if self.attempt_count < 3:
                    raise Exception("Transient error")
                return "Success after retry"

        provider = FailingProvider(llm="test-model")
        messages = [{"role": "user", "content": "Hello"}]

        response = provider.chat_completion(messages)
        assert response == "Success after retry"
        assert provider.attempt_count == 3

    def test_chat_completion_fails_after_max_retries(self):
        """Test: chat_completion raises error after all retries exhausted."""
        from llm_client.exceptions import ChatCompletionError

        class AlwaysFailingProvider(ConcreteProvider):
            def _chat_completion_impl(self, messages):
                raise Exception("Persistent error")

        provider = AlwaysFailingProvider(llm="test-model")
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(ChatCompletionError) as exc_info:
            provider.chat_completion(messages)

        assert "Persistent error" in str(exc_info.value)


class TestStreamingSupport:
    """Tests for streaming functionality."""

    def test_streaming_not_implemented_by_default(self):
        """Test: Streaming raises NotImplementedError by default."""
        from llm_client.exceptions import StreamingNotSupportedError

        provider = ConcreteProvider(llm="test-model")
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(StreamingNotSupportedError):
            list(provider.chat_completion_stream(messages))

    def test_streaming_can_be_implemented(self):
        """Test: Providers can implement streaming."""

        class StreamingProvider(ConcreteProvider):
            def _chat_completion_stream_impl(self, messages):
                yield "Hello"
                yield " "
                yield "world"

        provider = StreamingProvider(llm="test-model")
        messages = [{"role": "user", "content": "Hello"}]

        chunks = list(provider.chat_completion_stream(messages))
        assert chunks == ["Hello", " ", "world"]

    def test_streaming_wraps_errors(self):
        """Test: Streaming errors are wrapped in ChatCompletionError."""
        from llm_client.exceptions import ChatCompletionError

        class FailingStreamProvider(ConcreteProvider):
            def _chat_completion_stream_impl(self, messages):
                raise Exception("Stream error")

        provider = FailingStreamProvider(llm="test-model")
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(ChatCompletionError) as exc_info:
            list(provider.chat_completion_stream(messages))

        assert "Stream error" in str(exc_info.value)
