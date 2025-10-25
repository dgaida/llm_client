"""
utils.py
========
Hilfsfunktionen für das RAGChatbot_groq_API Notebook.

Dieses Modul enthält Dienstfunktionen zur Verarbeitung von PDF-Dateien
und zum Erstellen eines Vektorindexes mit Chroma und LlamaIndex.
"""

import os
import random
import chromadb
import requests
from typing import Any
from llama_index.readers.file.unstructured import UnstructuredReader
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

# Optionale Imports – nur wenn vorhanden
try:
    from chromadb.errors import ChromaError
except ImportError:
    ChromaError = Exception  # Fallback, falls Chroma nicht installiert ist

try:
    from openai import OpenAIError
except ImportError:
    OpenAIError = Exception  # Fallback für OpenAI nicht installiert


def read_pdf_files_with_unstructured_reader(pdf_directory: str = "pdfs") -> list[Document]:
    """
    Liest alle PDF-Dateien in einem angegebenen Verzeichnis mithilfe des `UnstructuredReader`
    aus LlamaIndex ein und gibt eine Liste von `Document`-Objekten zurück.

    Args:
        pdf_directory (str): Pfad zum Verzeichnis, das die PDF-Dateien enthält.
                             Standardmäßig wird das Unterverzeichnis 'pdfs' verwendet.

    Returns:
        list[Document]: Eine Liste von `Document`-Objekten, die den Inhalt der PDFs enthalten.

    Raises:
        FileNotFoundError: Wenn das angegebene Verzeichnis nicht existiert.
    """
    # --- Check if PDF directory exists ---
    if not os.path.exists(pdf_directory):
        raise FileNotFoundError(
            f"The folder '{pdf_directory}' does not exist. Please create it and add PDF files."
        )

    reader = UnstructuredReader()
    all_documents = []

    # Loop through PDF files in the folder and read each one individually
    for filename in os.listdir(pdf_directory):
        print("File:" + filename)
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)
            docs = reader.load_data(file_path)
            all_documents.extend(docs)

    return all_documents


def create_chromadb_vector_store_and_index(all_documents: list[Document]) -> VectorStoreIndex:
    """
    Erstellt einen temporären Chroma-Datenbank-Client und baut einen
    `VectorStoreIndex` aus den übergebenen Dokumenten auf.

    Bestehende Sammlungen (Collections) werden gelöscht, bevor eine neue erstellt wird.

    Args:
        all_documents (list[Document]): Eine Liste von Dokumenten, die in den Vektorspeicher
                                        aufgenommen werden sollen.

    Returns:
        VectorStoreIndex: Ein Vektorindex, der für semantische Suche und RAG-Anwendungen verwendet werden kann.
    """
    # Chroma will store vectors in a temporary database
    chroma_client = chromadb.EphemeralClient()

    # Get the current collections (if any) and clear them
    existing_collections = chroma_client.list_collections()
    # Loop through the collections and delete them by name
    for collection in existing_collections:
        # Extract the collection name and delete it
        collection_name = collection.name  # Assuming the 'Collection' object has a 'name' attribute
        chroma_client.delete_collection(collection_name)

    # add random number because deleting a collection not really deletes it -
    # this also does not help...
    chroma_collection = chroma_client.create_collection(f"dlml{random.randint(0, 100)}")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # here the documents are stored in the database
    index = VectorStoreIndex.from_documents(all_documents, storage_context=storage_context)

    return index


def safe_query_engine_call(query_engine: Any, query: str) -> str:
    """
    Führt eine Abfrage an einen LlamaIndex QueryEngine sicher aus und behandelt alle typischen Fehlerquellen.

    Args:
        query_engine (Any): Ein LlamaIndex QueryEngine-Objekt mit einer `query()`-Methode.
        query (str): Die Benutzeranfrage, die semantisch gesucht und vom LLM beantwortet werden soll.

    Returns:
        str: Der Antworttext des Modells oder eine standardisierte Fehlermeldung.
    """
    if not query or not isinstance(query, str):
        return "Error: Query must be a non-empty string."

    try:
        response = query_engine.query(query)
        return str(response)

    except (ValueError, TypeError) as e:
        print(f"[safe_query_engine_call] Input error: {e}")
        return "Invalid query input."

    except ChromaError as e:
        print(f"[safe_query_engine_call] ChromaDB error: {e}")
        return "Error accessing vector database."

    except OpenAIError as e:
        print(f"[safe_query_engine_call] LLM API error: {e}")
        return "Error communicating with language model provider."

    except requests.exceptions.RequestException as e:
        print(f"[safe_query_engine_call] Network error: {e}")
        return "Network connection problem during LLM request."

    except Exception as e:
        print(f"[safe_query_engine_call] Unexpected error: {e.__class__.__name__}: {e}")
        return "Unexpected error during inference."
