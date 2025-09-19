from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import json

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location) 

# ---------------------------
# Load documents from CSV & JSON
# ---------------------------
def load_csv_json():
    documents = []
    ids = []

    # Load JSON (if needed, just stored as raw string)
    with open("data.json") as f:
        json_data = json.load(f)
    documents.append(Document(
        page_content=json.dumps(json_data),
        metadata={"source": "data.json"}
    ))
    ids.append("json_0")

    # Load CSV
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.strip()
    for i, row in df.iterrows():
        documents.append(Document(
            page_content=f"{row['prefer_name']} {row['path']}",
            metadata={"type": row["type"], "data_type": row["data_type"], "source": "data.csv"}
        ))
        ids.append(f"csv_{i}")

    return documents, ids

# ---------------------------
# Load & chunk text documents
# ---------------------------
def load_txt(directory_path):
    raw_docs = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                raw_docs.append(Document(page_content=text, metadata={"source": filename}))
    return raw_docs

def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

def chunk_documents(raw_docs):
    chunked = []
    ids = []
    for idx, doc in enumerate(raw_docs):
        chunks = split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunked.append(Document(
                page_content=chunk,
                metadata={"source": doc.metadata["source"], "chunk": i+1}
            ))
            ids.append(f"txt_{idx}_{i}")
    return chunked, ids

# ---------------------------
# Build vector store
# ---------------------------
vector_store = Chroma(
    collection_name="api_reader",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    csv_json_docs, csv_json_ids = load_csv_json()
    txt_docs = load_txt("./doc")
    chunked_docs, txt_ids = chunk_documents(txt_docs)

    all_docs = csv_json_docs + chunked_docs
    all_ids = csv_json_ids + txt_ids

    print(f"==== Adding {len(all_docs)} documents to Chroma ====")
    vector_store.add_documents(documents=all_docs, ids=all_ids)

# ---------------------------
# Create retriever
# ---------------------------
retriever = vector_store.as_retriever(search_kwargs={"k": 10})
