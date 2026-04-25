"""Async support for LLM providers."""

from collections.abc import AsyncIterator
from typing import Any

from ..exceptions import (
    APIKeyNotFoundError,
    ChatCompletionError,
    ProviderNotAvailableError,
    StreamingNotSupportedError,
)
from ..utils.file_utils import detect_file_type, prepare_files_for_provider
from .base_provider import BaseProvider

# Optional imports
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore

try:
    from groq import APIStatusError, AsyncGroq
except ImportError:
    AsyncGroq = None  # type: ignore
    APIStatusError = None  # type: ignore


class AsyncProviderMixin:
    """Mixin for async provider functionality.

    This mixin adds async chat completion methods to providers.
    """

    async def achat_completion(self, messages: list[dict[str, str]]) -> str:
        """Execute an async chat completion request.

        Args:
            messages: List of message dictionaries.

        Returns:
            Generated text response.

        Raises:
            ChatCompletionError: If the API call fails.

        Examples:
            >>> messages = [{"role": "user", "content": "Hello"}]
            >>> response = await provider.achat_completion(messages)
        """
        try:
            return await self._achat_completion_impl(messages)
        except Exception as e:
            raise ChatCompletionError(self.__class__.__name__, e) from e

    async def _achat_completion_impl(self, messages: list[dict[str, str]]) -> str:
        """Internal implementation of async chat completion.

        Must be implemented by concrete providers.

        Args:
            messages: List of message dictionaries.

        Returns:
            Generated text response.
        """
        raise NotImplementedError("Async chat completion not implemented")

    async def achat_completion_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Execute async chat completion with tools."""
        try:
            return await self._achat_completion_with_tools_impl(messages, tools, tool_choice)
        except NotImplementedError as err:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support tool calling"
            ) from err
        except Exception as e:
            raise ChatCompletionError(self.__class__.__name__, e) from e

    async def _achat_completion_with_tools_impl(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Internal async implementation of tool calling."""
        raise NotImplementedError("Async tool calling not implemented")

    async def achat_completion_with_files(
        self,
        messages: list[dict[str, str]],
        files: list[str] | None = None,
    ) -> str:
        """Execute async chat completion with files."""
        try:
            return await self._achat_completion_with_files_impl(messages, files)
        except NotImplementedError as err:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support file uploads"
            ) from err
        except Exception as e:
            raise ChatCompletionError(self.__class__.__name__, e) from e

    async def _achat_completion_with_files_impl(
        self,
        messages: list[dict[str, str]],
        files: list[str] | None = None,
    ) -> str:
        """Internal async implementation of file upload."""
        raise NotImplementedError("Async file upload not implemented")

    async def achat_completion_stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Stream response tokens asynchronously.

        Args:
            messages: List of message dictionaries.

        Yields:
            Individual tokens or chunks of response text.

        Raises:
            StreamingNotSupportedError: If streaming not supported.
            ChatCompletionError: If the streaming API call fails.

        Examples:
            >>> messages = [{"role": "user", "content": "Tell me a story"}]
            >>> async for chunk in provider.achat_completion_stream(messages):
            ...     print(chunk, end="", flush=True)
        """
        try:
            async for chunk in self._achat_completion_stream_impl(messages):
                yield chunk
        except NotImplementedError as err:
            raise StreamingNotSupportedError(
                self.__class__.__name__, "Provider does not implement async streaming"
            ) from err
        except Exception as e:
            raise ChatCompletionError(self.__class__.__name__, e) from e

    async def _achat_completion_stream_impl(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Internal implementation of async streaming.

        Can be overridden by concrete providers.

        Args:
            messages: List of message dictionaries.

        Yields:
            Individual tokens or chunks.
        """
        raise NotImplementedError("Async streaming not implemented")
        # Make this a generator
        yield  # pragma: no cover


class AsyncOpenAIProvider(BaseProvider, AsyncProviderMixin):
    """Async OpenAI provider."""

    def _initialize_client(self, **kwargs: Any) -> None:
        """Initialize async OpenAI client."""
        if not self.is_available():
            raise ProviderNotAvailableError("openai", "openai")

        api_key = kwargs.get("api_key")
        if not api_key:
            raise APIKeyNotFoundError("openai", "OPENAI_API_KEY")

        self.client = AsyncOpenAI(api_key=api_key)

    def _chat_completion_impl(self, messages: list[dict[str, str]]) -> str:
        """Sync method raises error - use async version."""
        raise RuntimeError(
            "AsyncOpenAIProvider only supports async methods. " "Use achat_completion() instead."
        )

    async def _achat_completion_impl(self, messages: list[dict[str, str]]) -> str:
        """Execute async chat completion with OpenAI."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        response = await self.client.chat.completions.create(
            model=self.llm,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    async def _achat_completion_with_tools_impl(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Execute async chat completion with tools using OpenAI."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        kwargs = {
            "model": self.llm,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": tools,
        }

        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        response = await self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        return {
            "content": choice.message.content,
            "tool_calls": (
                [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in (choice.message.tool_calls or [])
                ]
                if choice.message.tool_calls
                else None
            ),
        }

    async def _achat_completion_with_files_impl(
        self,
        messages: list[dict[str, str]],
        files: list[str] | None = None,
    ) -> str:
        """Execute async chat completion with files using OpenAI."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        enhanced_messages = messages.copy()

        if files:
            file_data = prepare_files_for_provider(files, "openai")

            if enhanced_messages and enhanced_messages[-1]["role"] == "user":
                last_msg = enhanced_messages[-1]
                if isinstance(last_msg["content"], str):
                    text_content = last_msg["content"]
                    last_msg["content"] = [{"type": "text", "text": text_content}]

                last_msg["content"].extend(file_data)
            else:
                enhanced_messages.append({"role": "user", "content": file_data})

        response = await self.client.chat.completions.create(
            model=self.llm,
            messages=enhanced_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content

    async def _achat_completion_stream_impl(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Stream async chat completion with OpenAI."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        stream = await self.client.chat.completions.create(
            model=self.llm,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    @staticmethod
    def get_default_model() -> str:
        """Get default OpenAI model."""
        return "gpt-4o-mini"

    @staticmethod
    def is_available() -> bool:
        """Check if AsyncOpenAI is available."""
        return AsyncOpenAI is not None


class AsyncGroqProvider(BaseProvider, AsyncProviderMixin):
    """Async Groq provider."""

    def _initialize_client(self, **kwargs: Any) -> None:
        """Initialize async Groq client."""
        if not self.is_available():
            raise ProviderNotAvailableError("groq", "groq")

        api_key = kwargs.get("api_key")
        if not api_key:
            raise APIKeyNotFoundError("groq", "GROQ_API_KEY")

        self.client = AsyncGroq(api_key=api_key)

    def _chat_completion_impl(self, messages: list[dict[str, str]]) -> str:
        """Sync method raises error - use async version."""
        raise RuntimeError(
            "AsyncGroqProvider only supports async methods. " "Use achat_completion() instead."
        )

    async def _achat_completion_impl(self, messages: list[dict[str, str]]) -> str:
        """Execute async chat completion with Groq."""
        if not self.client:
            raise RuntimeError("Groq client not initialized")

        try:
            response = await self.client.chat.completions.create(
                model=self.llm,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            error_str = str(e)
            if ((hasattr(e, "status_code") and e.status_code == 413) or "413" in error_str) and (
                "rate_limit_exceeded" in error_str or "Rate limit exceeded" in error_str
            ):
                from .providers import GroqProvider

                # Reuse the fallback logic from the sync provider
                fallback_model = GroqProvider._find_fallback_model(None, error_str)
                if fallback_model:
                    self.llm = fallback_model
                    return await self._achat_completion_impl(messages)
            raise

        return response.choices[0].message.content

    async def _achat_completion_with_tools_impl(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Execute async chat completion with tools using Groq."""
        if not self.client:
            raise RuntimeError("Groq client not initialized")

        kwargs = {
            "model": self.llm,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": tools,
        }

        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        response = await self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        return {
            "content": choice.message.content,
            "tool_calls": (
                [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in (choice.message.tool_calls or [])
                ]
                if choice.message.tool_calls
                else None
            ),
        }

    async def _achat_completion_with_files_impl(
        self,
        messages: list[dict[str, str]],
        files: list[str] | None = None,
    ) -> str:
        """Execute async chat completion with files using Groq."""
        if not self.client:
            raise RuntimeError("Groq client not initialized")

        if files:
            for file_path in files:
                if detect_file_type(file_path) != "image":
                    raise ValueError("Groq only supports image files for vision models")

        enhanced_messages = messages.copy()

        if files:
            file_data = prepare_files_for_provider(files, "groq")

            if enhanced_messages and enhanced_messages[-1]["role"] == "user":
                last_msg = enhanced_messages[-1]
                if isinstance(last_msg["content"], str):
                    text_content = last_msg["content"]
                    last_msg["content"] = [{"type": "text", "text": text_content}]

                last_msg["content"].extend(file_data)
            else:
                enhanced_messages.append({"role": "user", "content": file_data})

        response = await self.client.chat.completions.create(
            model=self.llm,
            messages=enhanced_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content

    async def _achat_completion_stream_impl(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Stream async chat completion with Groq."""
        if not self.client:
            raise RuntimeError("Groq client not initialized")

        stream = await self.client.chat.completions.create(
            model=self.llm,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    @staticmethod
    def get_default_model() -> str:
        """Get default Groq model."""
        return "qwen/qwen3-32b"

    @staticmethod
    def is_available() -> bool:
        """Check if AsyncGroq is available."""
        return AsyncGroq is not None


class AsyncGeminiProvider(BaseProvider, AsyncProviderMixin):
    """Async Gemini provider via OpenAI compatibility."""

    def _initialize_client(self, **kwargs: Any) -> None:
        """Initialize async Gemini client."""
        if not self.is_available():
            raise ProviderNotAvailableError("gemini", "openai")

        api_key = kwargs.get("api_key")
        if not api_key:
            raise APIKeyNotFoundError("gemini", "GEMINI_API_KEY")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def _chat_completion_impl(self, messages: list[dict[str, str]]) -> str:
        """Sync method raises error - use async version."""
        raise RuntimeError(
            "AsyncGeminiProvider only supports async methods. " "Use achat_completion() instead."
        )

    async def _achat_completion_impl(self, messages: list[dict[str, str]]) -> str:
        """Execute async chat completion with Gemini."""
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        response = await self.client.chat.completions.create(
            model=self.llm,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    async def _achat_completion_with_tools_impl(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict | None = None,
    ) -> dict:
        """Execute async chat completion with tools using Gemini."""
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        kwargs = {
            "model": self.llm,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": tools,
        }

        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        response = await self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        return {
            "content": choice.message.content,
            "tool_calls": (
                [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in (choice.message.tool_calls or [])
                ]
                if choice.message.tool_calls
                else None
            ),
        }

    async def _achat_completion_with_files_impl(
        self,
        messages: list[dict[str, str]],
        files: list[str] | None = None,
    ) -> str:
        """Execute async chat completion with files using Gemini."""
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        enhanced_messages = messages.copy()

        if files:
            file_data = prepare_files_for_provider(files, "gemini")

            if enhanced_messages and enhanced_messages[-1]["role"] == "user":
                last_msg = enhanced_messages[-1]
                if isinstance(last_msg["content"], str):
                    text_content = last_msg["content"]
                    last_msg["content"] = [{"type": "text", "text": text_content}]

                last_msg["content"].extend(file_data)
            else:
                enhanced_messages.append({"role": "user", "content": file_data})

        response = await self.client.chat.completions.create(
            model=self.llm,
            messages=enhanced_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content

    async def _achat_completion_stream_impl(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Stream async chat completion with Gemini."""
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        stream = await self.client.chat.completions.create(
            model=self.llm,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    @staticmethod
    def get_default_model() -> str:
        """Get default Gemini model."""
        return "gemini-3.1-flash-lite-preview"

    @staticmethod
    def is_available() -> bool:
        """Check if AsyncOpenAI (for Gemini) is available."""
        return AsyncOpenAI is not None
