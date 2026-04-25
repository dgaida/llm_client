from llm_client import LLMClient


def main():
    """Demonstriert die Verwendung von LLMClient mit verschiedenen APIs."""

    # Beispiel 1: Automatische API-Auswahl
    print("=" * 50)
    print("Beispiel 1: Automatische API-Auswahl")
    print("=" * 50)
    client = LLMClient()
    print(f"Verwendete API: {client.api_choice}")
    print(f"Verwendetes Modell: {client.llm}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Erkläre kurz, was ein neuronales Netz ist."},
    ]

    try:
        response = client.chat_completion(messages)
        print("\nAntwort:\n", response)
    except Exception as e:
        print(f"Fehler: {e}")

    # Beispiel 2: Gemini explizit nutzen (falls API Key vorhanden)
    print("\n" + "=" * 50)
    print("Beispiel 2: Google Gemini")
    print("=" * 50)
    try:
        gemini_client = LLMClient(
            api_choice="gemini", llm="gemini-3.1-flash-lite-preview", temperature=0.7
        )
        print(f"Verwendetes Modell: {gemini_client.llm}")

        gemini_messages = [{"role": "user", "content": "Erkläre Quantencomputing in einem Satz."}]

        gemini_response = gemini_client.chat_completion(gemini_messages)
        print("\nAntwort:\n", gemini_response)
    except RuntimeError as e:
        print(f"Gemini nicht verfügbar: {e}")
        print("Tipp: Setze GEMINI_API_KEY in secrets.env")
    except Exception as e:
        print(f"Fehler: {e}")

    # Beispiel 3: Vergleich verschiedener Modelle
    print("\n" + "=" * 50)
    print("Beispiel 3: API-Vergleich (falls Keys vorhanden)")
    print("=" * 50)

    question = "Was ist der Unterschied zwischen KI und ML?"
    apis_to_test = [
        ("openai", "gpt-4o-mini"),
        ("groq", "qwen/qwen3-32b"),
        ("gemini", "gemini-3.1-flash-lite-preview"),
    ]

    for api_name, model_name in apis_to_test:
        try:
            test_client = LLMClient(api_choice=api_name, llm=model_name)
            print(f"\n{api_name.upper()} ({model_name}):")
            result = test_client.chat_completion([{"role": "user", "content": question}])
            print(result[:200] + "..." if len(result) > 200 else result)
        except Exception as e:
            print(f"{api_name.upper()}: Nicht verfügbar ({type(e).__name__})")


if __name__ == "__main__":
    main()
