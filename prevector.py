from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# ---------------------------
# Load & preprocess documents
# ---------------------------
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(Document(
                    page_content=text,
                    metadata={"source": filename}
                ))
    return documents

def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

directory_path = "./doc"
raw_documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(raw_documents)} documents")

chunked_documents = []
for doc in raw_documents:
    chunks = split_text(doc.page_content)
    for i, chunk in enumerate(chunks):
        chunked_documents.append(Document(
            page_content=chunk,
            metadata={"source": doc.metadata["source"], "chunk": i+1}
        ))

# ---------------------------
# Create Chroma Vector Store
# ---------------------------
vector_store = Chroma(
    collection_name="api_reader",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    print("==== Adding documents to Chroma ====")
    vector_store.add_documents(chunked_documents)

retriever = vector_store.as_retriever(search_kwargs={"k": 10})
