"""
llm_client
==========

Ein universelles Interface für LLM-Zugriffe (OpenAI, Groq, Ollama).

Dieses Package bietet die Klasse `LLMClient`, die automatisch erkennt,
welche API verfügbar ist (basierend auf `secrets.env`) und entsprechend
die Methode `chat_completion()` aufruft.
"""

from .llm_client import LLMClient
from .adapter import LLMClientAdapter

__all__ = ["LLMClient", "LLMClientAdapter"]


__version__ = "0.1.0"
__author__ = "Daniel Gaida"
__license__ = "MIT"
