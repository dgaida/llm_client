# RAG Chatbot with Groq API and Text-to-Speech (TTS)

[![Open In Colab](../../assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/llm_client/blob/master/notebooks/RAGChatbot_groq_API_t2s.ipynb)

- [Overview of Retrieval-Augmented Generation (RAG)](#overview-of-retrieval-augmented-generation-rag)  
- [🚀 Notebook Content](#notebook-content)  
- [🔑 Required API Keys](#required-api-keys)  
- [🦮 Creating a Hugging Face Access Token](#creating-a-hugging-face-access-token)  
- [⚡️ Creating a Groq API Key](#creating-a-groq-api-key)  
- [🔮 Creating an OpenAI API Key](#creating-an-openai-api-key)  
- [Creating a Google Gemini API Key](#creating-a-google-gemini-api-key)  
- [☁️ Store API Keys as Secrets in Google Colab](#store-api-keys-as-secrets-in-google-colab)  
- [⚙️ Using LLMClient in the Notebook](#using-llmclient-in-the-notebook)  
- [Resources for RAG](#resources-for-rag)  
- [🧩 License](#license)  

The notebook [`RAGChatbot_groq_API_t2s.ipynb`](https://github.com/dgaida/llm_client/blob/master/notebooks/RAGChatbot_groq_API_t2s.ipynb) shows how to create a **Retrieval-Augmented Generation (RAG)** chatbot using the [`LLMClient`](https://github.com/dgaida/llm_client/blob/master/llm_client/llm_client.py) class, which also features a **Text-to-Speech (TTS)** function. The [**Kokoro**](https://huggingface.co/hexgrad/Kokoro-82M) model is used for speech synthesis.

![RAG-Chatbot GUI](../../assets/tutorials/PDF_RAG_Chatbot.png){ width="750" style="display: block; margin: 0 auto" }

---

## Overview of Retrieval-Augmented Generation (RAG) {: #overview-of-retrieval-augmented-generation-rag }

Retrieval-Augmented Generation (RAG) combines **knowledge from your own documents** with the **linguistic competence of large AI models** like ChatGPT.
Instead of the model only accessing its internal (and limited) training knowledge, RAG first searches specifically in a **knowledge base** or **document collection** for relevant text passages ("Retrieval") and then passes these along with the user query to the **Large Language Model** ("Generation").

Additionally, this tutorial integrates **Text-to-Speech (TTS)** to convert the generated answers directly into speech. This allows for a more natural interaction with the chatbot.

The following diagram shows the basic structure of a RAG system:

![High-level overview of the Retrieval Augmented Generation System](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*ys44J6jLm5vSTjFIMDDEfw.png)

*Figure: "High-level overview of the Retrieval Augmented Generation System"
by [Maanjunath S Naragund](https://maanjunathn07ds.medium.com/), taken from [this blog post on Medium](https://blog.gopenai.com/step-by-step-guide-to-implementing-retrieval-augmented-generation-in-python-4801be2771c3).
Icons by Flaticon. Used under the right of quotation (§ 51 UrhG). This figure is **not under the MIT license** of this repository.*

---

## 🚀 Notebook Content {: #notebook-content }

The notebook demonstrates:

1. Installation of required packages in **Google Colab** (including `kokoro` for TTS)  
2. Using the `LLMClient` class for text generation  
3. Building a **RAG workflow** with PDF documents and ChromaDB  
4. Integration of **Text-to-Speech (TTS)** with the Kokoro model for speech output of responses  

---

## 🔑 Required API Keys {: #required-api-keys }

| Service | Required | Purpose |
|--------|----------|--------|
| **Hugging Face Access Token** | ✅ **required** | Download the embedding model and the Kokoro TTS model |
| **Groq API Key** | optional | Use the [`Groq`](https://console.groq.com/home) LLM API |
| **OpenAI API Key** | optional | Use the OpenAI LLM API |

---

## 🦮 Creating a Hugging Face Access Token {: #creating-a-hugging-face-access-token }

The Hugging Face Access Token is required to access **embedding models** and other AI models from the Hugging Face Model Hub, which are used to calculate sentence embeddings. These are downloaded from the Model Hub and executed locally.

1. Create a free account at [https://huggingface.co/](https://huggingface.co/) or log in (if necessary).  

2. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)  

![Hugging Face – Settings Menu](../../assets/tutorials/Hugging_Face_settings_menu_access_tokens.png){ width="250" style="display: block; margin: 0 auto" }

3. Click on the **"Create new token"** button  

![Hugging Face – User Access Tokens](../../assets/tutorials/Hugging_Face_User_Access_Tokens.png){ width="850" style="display: block; margin: 0 auto" }

4. Enter a name (e.g., `colab-rag`) and select **Type: Write**  

![Hugging Face – Create New Write Token](../../assets/tutorials/Hugging_Face_create_new_write_token.png){ width="850" style="display: block; margin: 0 auto" }

5. Copy the displayed token (usually starts with `hf_...`).  

---

## ⚡️ Creating a Groq API Key {: #creating-a-groq-api-key }

The Groq API Key allows access to publicly available **LLMs** that can be used for particularly fast **text generation and question answering** in the RAG workflow. These LLMs are executed in the GroqCloud.

1. Create a free account at [https://groq.com/](https://groq.com/) or log in (if necessary).  
2. Visit [https://console.groq.com/keys](https://console.groq.com/keys)  
3. Click on **"Create API Key"**  

![Groq API Keys – Create API Key](../../assets/tutorials/groq_API_Keys_Create_API_Key.png)

4. Copy the key (usually starts with `groq_...`).  

---

## 🔮 Creating an OpenAI API Key {: #creating-an-openai-api-key }

The OpenAI API Key allows the use of **OpenAI models** (e.g., GPT-4 or GPT-4o) to generate **context-related answers** in the Retrieval-Augmented Generation system.

1. Log in to [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)  

![OpenAI API – API Keys](../../assets/tutorials/OpenAI_API_API_keys.png){ width="175" style="display: block; margin: 0 auto" }

2. Click on "Create new secret key"  

![OpenAI API – Create New Secret Key](../../assets/tutorials/OpenAI_API_Create_new_secret_key.png){ width="450" style="display: block; margin: 0 auto" }

3. Copy the key (usually starts with `sk-...`).  

---

## Creating a Google Gemini API Key {: #creating-a-google-gemini-api-key }

1. Visit [Google AI Studio](https://aistudio.google.com/apikey)  
2. Click on **"Get API Key"** or **"Create API Key"**  
3. Select a Google Cloud project or create a new one  
4. Copy the generated API key (starts with `AIzaSy...`)  

**Note**: The Gemini API is accessed via the OpenAI compatibility mode, therefore only the `openai` Python package is required.

---

## ☁️ Store API Keys as Secrets in Google Colab {: #store-api-keys-as-secrets-in-google-colab }

1. Click on the key symbol 🔑 in the menu on the left  

![Google Colab – Secrets – API Keys](../../assets/tutorials/Google_Colab_secrets_api_keys.png){ width="600" style="display: block; margin: 0 auto" }

2. Create the following secrets:  

   | Name | Value |
   |-------|------|
   | `HF_TOKEN` | your Hugging Face Access Token |
   | `GROQ_API_KEY` | (optional) your Groq API Key |
   | `OPENAI_API_KEY` | (optional) your OpenAI API Key |

---

## ⚙️ Using LLMClient in the Notebook {: #using-llmclient-in-the-notebook }

```python
from llm_client import LLMClient

# LLMClient automatically detects which keys are set
client = LLMClient()

print("Used API:", client.api_choice)
print("Model:", client.llm)
```

---

## Resources for RAG

[`Coursera Course on Retrieval Augmented Generation (RAG) by DeepLearning.AI`](https://www.coursera.org/learn/retrieval-augmented-generation-rag/)

---

## 🧩 License {: #license }

This notebook is part of the repository [**dgaida/llm_client**](https://github.com/dgaida/llm_client).
© 2025 – Daniel Gaida, Technical University.
Licensed under the **MIT License**.
