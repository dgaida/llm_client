from llm_client import LLMClient


def main():
    # Beispiel: Instanziiere die Klasse (automatische API-Auswahl)
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


if __name__ == "__main__":
    main()
