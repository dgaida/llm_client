# 🧠 RAG Chatbot mit LLMClient (Groq, OpenAI & Hugging Face)

Dieses Notebook zeigt, wie man mit der Klasse [`LLMClient`](../llm_client/llm_client.py)  
einen **Retrieval-Augmented-Generation (RAG)**-Chatbot erstellt, der wahlweise
über **Groq**, **OpenAI** oder **Ollama** betrieben wird.  

---

## 🚀 Inhalt des Notebooks

Das Notebook demonstriert:

1. Installation der benötigten Packages in **Google Colab**
2. Nutzung der `LLMClient`-Klasse mit:
   - 🧩 **Groq API** *(optional)*
   - 🔮 **OpenAI API** *(optional)*
   - 💻 **Ollama (local)** *(Fallback)*
3. Aufbau eines einfachen **RAG-Workflows**:
   - Dokumente laden  
   - Embeddings mit einem Embedding-Modell von **Hugging Face** erzeugen  
   - Antworten aus LLM + Vektorstore kombinieren

---

## 🔑 Erforderliche API Keys

| Dienst | Pflicht | Zweck |
|--------|----------|--------|
| **Hugging Face API Key** | ✅ **erforderlich** | Laden des Embedding-Modells |
| **Groq API Key** | optional | Nutzung der Groq LLM-API |
| **OpenAI API Key** | optional | Nutzung der OpenAI LLM-API |

Wenn weder Groq- noch OpenAI-Key gesetzt sind, nutzt `LLMClient` automatisch **Ollama** (funktioniert nur lokal und nicht in Google Colab).

---

## 🦮 Hugging Face API Key erstellen

1. Gehe zu [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)  
   ![Hugging Face – Settings Menu](images/Hugging%20Face%20-%20settings%20menu%20-%20access%20tokens.png)

2. Klicke auf **„New token“**  
   ![Hugging Face – Create New Write Token](images/Hugging%20Face%20-%20create%20new%20write%20token.png)

3. Gib einen Namen ein (z. B. `colab-rag`) und wähle **Type: Write**  
4. Kopiere den angezeigten Token (beginnt meist mit `hf_...`).

---

## ⚡️ Groq API Key erstellen

1. Besuche [https://console.groq.com/keys](https://console.groq.com/keys)  
2. Klicke auf **„Create API Key“**  
   ![Groq API Keys – Create API Key](images/groq%20API%20Keys%20-%20Create%20API%20Key.png)
3. Kopiere den Schlüssel (beginnt meist mit `groq_...`).

---

## 🔮 OpenAI API Key erstellen

1. Melde dich bei [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys) an  
   ![OpenAI API – API Keys](images/OpenAI%20API%20-%20API%20keys.png)

2. Klicke auf **„Create new secret key“**  
   ![OpenAI API – Create New Secret Key](images/OpenAI%20API%20-%20Create%20new%20secret%20key.png)

3. Kopiere den Key (beginnt meist mit `sk-...`).

---

## ☁️ API Keys als Secrets in Google Colab hinterlegen

1. Öffne in Colab **„Datei > Einstellungen > Secrets“**  
   oder klicke im Menü links auf das Schlüssel-Symbol 🔑  
   ![Google Colab – Secrets – API Keys](images/Google%20Colab%20-%20secrets%20-%20api%20keys.png)

2. Lege folgende Secrets an:

   | Name | Wert |
   |-------|------|
   | `HF_TOKEN` | dein Hugging Face Token |
   | `GROQ_API_KEY` | (optional) dein Groq API Key |
   | `OPENAI_API_KEY` | (optional) dein OpenAI API Key |

---

## ⚙️ Nutzung im Notebook

```python
from llm_client import LLMClient

# LLMClient erkennt automatisch, welche Keys gesetzt sind
client = LLMClient()

print("Verwendete API:", client.api_choice)
print("Modell:", client.llm)
```

Falls kein Groq- oder OpenAI-Key gefunden wird, fällt der Client automatisch auf **Ollama** zurück (lokaler Betrieb).

---

## 🧩 Lizenz

Dieses Notebook ist Teil des Repositories [**dgaida/llm_client**](https://github.com/dgaida/llm_client).  
© 2025 – Daniel Gaida, Technische Hochschule.  
Lizenziert unter der **MIT License**.
