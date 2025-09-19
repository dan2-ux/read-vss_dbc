import json
import os
import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

db_location = "./chroma_langchain_db1"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    with open("data.json") as f:
        json_data = json.load(f)

    for i, item in enumerate(json_data):
        inp = item.get("input", "")
        out = item.get("output", {})
        api = out.get("api", "")
        val = out.get("value", "")

        page_content = f"Input: {inp} | API: {api} | Value: {val}"

        doc_id = f"data.json_{i}"
        document = Document(
            page_content=page_content,
            metadata={"api": api, "source": "data.json"},
            id=doc_id
        )
        ids.append(doc_id)
        documents.append(document)

vector_store = Chroma(
    collection_name="api_reader",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
