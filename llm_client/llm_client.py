"""LLM Client Module für universelle LLM-API Zugriffe."""

import os
from typing import Any, Literal

from dotenv import load_dotenv

# Optionale Imports – falls nicht installiert, wird das erkannt
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore


class LLMClient:
    """Eine universelle Klasse zur Nutzung von OpenAI, Groq oder Ollama.

    Diese Klasse erkennt automatisch verfügbare API-Keys und wählt die
    entsprechende API oder erlaubt manuelle Steuerung per Parameter.

    Attributes:
        api_choice: Die gewählte API ('openai', 'groq' oder 'ollama').
        llm: Name des verwendeten Modells.
        temperature: Sampling-Temperatur für die Generierung.
        max_tokens: Maximale Anzahl zu generierender Tokens.
        keep_alive: Ollama-spezifisch - wie lange Modell im Speicher bleibt.
        client: Instanz des gewählten API-Clients.
        openai_api_key: OpenAI API Key (falls vorhanden).
        groq_api_key: Groq API Key (falls vorhanden).

    Examples:
        >>> # Automatische API-Auswahl basierend auf verfügbaren Keys
        >>> client = LLMClient()
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response = client.chat_completion(messages)

        >>> # Manuell Ollama wählen
        >>> client = LLMClient(api_choice="ollama", llm="llama3.2:1b")
    """

    def __init__(
        self,
        llm: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_choice: Literal["openai", "groq", "ollama"] | None = None,
        secrets_path: str = "secrets.env",
        keep_alive: str = "5m",
    ) -> None:
        """Initialisiert den LLM Client.

        Args:
            llm: Name des Modells. Wenn None, wird ein Default-Modell gewählt.
            temperature: Sampling-Temperatur (0.0 bis 2.0). Standard: 0.7.
            max_tokens: Maximale Anzahl zu generierender Tokens. Standard: 512.
            api_choice: Explizite API-Wahl ('openai', 'groq', 'ollama').
                Wenn None, wird automatisch gewählt.
            secrets_path: Pfad zur secrets.env-Datei. Standard: "secrets.env".
            keep_alive: Ollama-Parameter für Modell-Caching. Standard: "5m".

        Raises:
            ValueError: Wenn api_choice einen ungültigen Wert hat.

        Examples:
            >>> client = LLMClient(llm="gpt-4o", temperature=0.5)
            >>> client = LLMClient(api_choice="ollama", max_tokens=1024)
        """
        # 1. Lade secrets.env, falls vorhanden
        if os.path.exists(secrets_path):
            load_dotenv(secrets_path)

        self.openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
        self.groq_api_key: str | None = os.getenv("GROQ_API_KEY")

        # 2. Fallback für Google Colab
        if self.openai_api_key is None or self.groq_api_key is None:
            try:
                import sys

                if "google.colab" in sys.modules or "COLAB_GPU" in os.environ:
                    # Google Colab erkannt
                    from google.colab import userdata

                    if self.openai_api_key is None:
                        self.openai_api_key = userdata.get("OPENAI_API_KEY")
                    if self.groq_api_key is None:
                        self.groq_api_key = userdata.get("GROQ_API_KEY")
            except ImportError:
                # Nicht auf Colab oder userdata nicht verfügbar
                pass

        # 3. Automatische API-Auswahl
        if api_choice is None:
            if self.openai_api_key:
                self.api_choice: str = "openai"
            elif self.groq_api_key:
                self.api_choice = "groq"
            else:
                self.api_choice = "ollama"
        else:
            valid_choices = {"openai", "groq", "ollama"}
            if api_choice.lower() not in valid_choices:
                raise ValueError(
                    f"Invalid api_choice: {api_choice}. " f"Must be one of {valid_choices}"
                )
            self.api_choice = api_choice.lower()

        # 4. Default-Modellauswahl
        if llm:
            self.llm: str = llm
        else:
            if self.api_choice == "openai":
                self.llm = "gpt-4o-mini"
            elif self.api_choice == "groq":
                self.llm = "moonshotai/kimi-k2-instruct-0905"
            else:
                self.llm = "llama3.2:1b"

        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.keep_alive: str = keep_alive

        # 5. Clients vorbereiten
        self.client: Any | None = None
        if self.api_choice == "openai" and OpenAI:
            self.client = OpenAI(api_key=self.openai_api_key)
        elif self.api_choice == "groq" and Groq:
            self.client = Groq(api_key=self.groq_api_key)

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        """Führt eine Chat-Completion mit der gewählten API aus.

        Args:
            messages: Liste von Nachrichten im Chat-Format.
                Jede Nachricht ist ein Dict mit 'role' und 'content' Keys.
                Beispiel: [{"role": "user", "content": "Hello!"}]

        Returns:
            Der generierte Text als String.

        Raises:
            RuntimeError: Wenn der gewählte Client nicht verfügbar ist.
            ValueError: Wenn api_choice ungültig ist.

        Examples:
            >>> client = LLMClient()
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "Explain AI."}
            ... ]
            >>> response = client.chat_completion(messages)
            >>> print(response)
        """
        if self.api_choice == "openai":
            if not self.client:
                raise RuntimeError("OpenAI client not available or not installed.")
            response = self.client.chat.completions.create(
                model=self.llm,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

        elif self.api_choice == "groq":
            if not self.client:
                raise RuntimeError("Groq client not available or not installed.")
            response = self.client.chat.completions.create(
                model=self.llm,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

        elif self.api_choice == "ollama":
            if not ollama:
                raise RuntimeError(
                    "Ollama Python package not available. "
                    "Please install it via `pip install ollama`."
                )
            response = ollama.chat(
                model=self.llm,
                messages=messages,
                stream=False,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "repeat_penalty": 1.2,
                    "top_k": 10,
                    "top_p": 0.5,
                },
                keep_alive=self.keep_alive,
            )
            return response["message"]["content"]

        else:
            raise ValueError(f"Unsupported API choice: {self.api_choice}")

    def __repr__(self) -> str:
        """Gibt eine String-Repräsentation des Clients zurück.

        Returns:
            String-Repräsentation mit API und Modell-Info.
        """
        return (
            f"LLMClient(api={self.api_choice}, model={self.llm}, "
            f"temperature={self.temperature})"
        )
