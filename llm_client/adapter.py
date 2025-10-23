from typing import List, Optional
from pydantic import Field
from llama_index.core.llms import LLM, ChatMessage, ChatResponse, CompletionResponse, LLMMetadata
from .llm_client import LLMClient


class LLMClientAdapter(LLM):
    """
    Adapter für llama_index, um den LLMClient als normales LLM zu verwenden.
    """

    client: Optional[LLMClient] = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)

    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        # Konvertiere llama_index Nachrichten in dict
        hf_messages = [{"role": m.role, "content": m.content} for m in messages]

        # Nutze LLMClient
        response = self.client.chat_completion(hf_messages)

        return ChatResponse(
            message=ChatMessage(role="assistant", content=response)
        )

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        raise NotImplementedError("complete not implemented")

    def stream_chat(self, *args, **kwargs):
        raise NotImplementedError("stream_chat not implemented")

    def stream_complete(self, *args, **kwargs):
        raise NotImplementedError("stream_complete not implemented")

    async def astream_chat(self, *args, **kwargs):
        raise NotImplementedError("astream_chat not implemented")

    async def astream_complete(self, *args, **kwargs):
        raise NotImplementedError("astream_complete not implemented")

    async def achat(self, *args, **kwargs):
        raise NotImplementedError("achat not implemented")

    async def acomplete(self, *args, **kwargs):
        raise NotImplementedError("acomplete not implemented")

    @property
    def model(self) -> str:
        return self.client.llm

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=2048,
            num_output=512,
            is_chat_model=True,
            model_name=self.model
        )
