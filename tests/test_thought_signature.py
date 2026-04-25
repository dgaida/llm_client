import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from llm_client.providers.providers import GeminiProvider
import asyncio
from llm_client.providers.async_providers import AsyncGeminiProvider
import logging

class TestThoughtSignature(unittest.TestCase):
    def setUp(self):
        self.patcher = patch('llm_client.providers.providers.OpenAI')
        self.mock_openai = self.patcher.start()
        self.mock_client = MagicMock()
        self.mock_openai.return_value = self.mock_client

        self.provider = GeminiProvider(llm="gemini-3.1-flash-lite-preview", api_key="fake-key")

        self.async_patcher = patch('llm_client.providers.async_providers.AsyncOpenAI')
        self.mock_async_openai = self.async_patcher.start()
        self.mock_async_client = MagicMock()
        self.mock_async_client.chat = MagicMock()
        self.mock_async_client.chat.completions = MagicMock()
        self.mock_async_client.chat.completions.create = AsyncMock()
        self.mock_async_openai.return_value = self.mock_async_client

        self.async_provider = AsyncGeminiProvider(llm="gemini-3.1-flash-lite-preview", api_key="fake-key")

    def tearDown(self):
        self.patcher.stop()
        self.async_patcher.stop()

    def test_sync_tool_call_thought_signature_preserved(self):
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "Berlin"}'
        mock_tool_call.extra_content = {"google": {"thought_signature": "sig_abc_123"}}

        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_choice.message.tool_calls = [mock_tool_call]

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        self.mock_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "How is the weather?"}]
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]

        result = self.provider.chat_completion_with_tools(messages, tools)

        tool_calls = result.get("tool_calls", [])
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["extra_content"], {"google": {"thought_signature": "sig_abc_123"}})

    def test_sync_chat_logging(self):
        mock_message = MagicMock()
        mock_message.content = "Berlin is sunny."
        mock_message.extra_content = {"google": {"thought_signature": "sig_chat_123"}}

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        self.mock_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "How is the weather in Berlin?"}]

        with self.assertLogs('llm_client.providers.providers', level='DEBUG') as cm:
            result = self.provider.chat_completion(messages)
            self.assertEqual(result, "Berlin is sunny.")
            self.assertTrue(any("Received extra_content" in output for output in cm.output))
            self.assertTrue(any("sig_chat_123" in output for output in cm.output))

    def test_async_tool_call_thought_signature_preserved(self):
        async def run_test():
            mock_tool_call = MagicMock()
            mock_tool_call.id = "call_123"
            mock_tool_call.type = "function"
            mock_tool_call.function.name = "get_weather"
            mock_tool_call.function.arguments = '{"location": "Berlin"}'
            mock_tool_call.extra_content = {"google": {"thought_signature": "sig_async_abc"}}

            mock_choice = MagicMock()
            mock_choice.message.content = None
            mock_choice.message.tool_calls = [mock_tool_call]

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]

            # Async mock
            self.mock_async_client.chat.completions.create.return_value = mock_response

            messages = [{"role": "user", "content": "How is the weather?"}]
            tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]

            result = await self.async_provider.achat_completion_with_tools(messages, tools)

            tool_calls = result.get("tool_calls", [])
            self.assertEqual(len(tool_calls), 1)
            self.assertEqual(tool_calls[0]["extra_content"], {"google": {"thought_signature": "sig_async_abc"}})

        asyncio.run(run_test())

    def test_async_chat_logging(self):
        async def run_test():
            mock_message = MagicMock()
            mock_message.content = "Berlin is sunny."
            mock_message.extra_content = {"google": {"thought_signature": "sig_async_chat_123"}}

            mock_choice = MagicMock()
            mock_choice.message = mock_message

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]

            # Async mock
            self.mock_async_client.chat.completions.create.return_value = mock_response

            messages = [{"role": "user", "content": "How is the weather in Berlin?"}]

            with self.assertLogs('llm_client.providers.providers', level='DEBUG') as cm:
                result = await self.async_provider.achat_completion(messages)
                self.assertEqual(result, "Berlin is sunny.")
                self.assertTrue(any("Received extra_content" in output for output in cm.output))
                self.assertTrue(any("sig_async_chat_123" in output for output in cm.output))

        asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main()
