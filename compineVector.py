from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import json

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location) 

if add_documents:
    documents = []
    ids = []
    with open("data.json") as f:
        json_data = json.load(f)
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip()
    databases = ["data.csv", "data.json"]

    for database in databases:
        if database.endswith(".csv"):
            for i, row in df.iterrows():
                document = Document(
                    page_content= str(row["prefer_name"]) + " " + str(row["path"]),
                    metadata={"type": row["type"], "data_type": row["data_type"]},
                    id=str(i)
                )
                ids.append(str(i))
                documents.append(document)
        if database.endswith(".json"):
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
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 10}
)