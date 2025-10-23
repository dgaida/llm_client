import os
from dotenv import load_dotenv

# Optionale Imports – falls nicht installiert, wird das erkannt
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import ollama
except ImportError:
    ollama = None


class LLMClient:
    """
    Eine universelle Klasse zur Nutzung von OpenAI, Groq oder Ollama.
    Erkennt automatisch, welche API-Keys vorhanden sind,
    oder erlaubt manuelle Steuerung per Parameter.
    """

    def __init__(
        self,
        llm: str = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_choice: str = None,
        secrets_path: str = "secrets.env",
        keep_alive: str = "5m",
    ):
        """
        Parameter:
        -----------
        llm : str
            Name des Modells (z. B. 'gpt-4o-mini', 'meta-llama/llama-guard-4-12b', 'llama3.1')
        temperature : float
            Sampling-Temperatur (Standard: 0.7)
        max_tokens : int
            Maximale Tokenanzahl in der Antwort (Standard: 512)
        api_choice : str
            'openai', 'groq' oder 'ollama'. Wenn None, wird anhand vorhandener API Keys entschieden.
        secrets_path : str
            Pfad zur secrets.env-Datei
        keep_alive : str
            Ollama Parameter – wie lange das Modell im Speicher bleiben soll.
        """
        load_dotenv(secrets_path)

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        # Automatische API-Auswahl
        if api_choice is None:
            if self.openai_api_key:
                self.api_choice = "openai"
            elif self.groq_api_key:
                self.api_choice = "groq"
            else:
                self.api_choice = "ollama"
        else:
            self.api_choice = api_choice.lower()

        # Default-Modellauswahl
        if llm:
            self.llm = llm
        else:
            if self.api_choice == "openai":
                self.llm = "gpt-4o-mini"
            elif self.api_choice == "groq":
                self.llm = "moonshotai/kimi-k2-instruct-0905"
            else:
                self.llm = "llama3.2:1b"

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive

        # Clients vorbereiten
        self.client = None
        if self.api_choice == "openai" and OpenAI:
            self.client = OpenAI(api_key=self.openai_api_key)
        elif self.api_choice == "groq" and Groq:
            self.client = Groq(api_key=self.groq_api_key)

    def chat_completion(self, messages):
        """
        Führt eine Chat-Completion mit der gewählten API aus.

        Parameter:
        -----------
        messages : list[dict]
            Liste von Nachrichten im Format [{"role": "user", "content": "..."}]
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
                    "Ollama Python package not available. Please install it via `pip install ollama`."
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
