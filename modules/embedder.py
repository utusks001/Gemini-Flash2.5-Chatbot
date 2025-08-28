# Embedding & vector store
# modules/embedder.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def build_chroma_from_documents(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Gunakan Chroma default tanpa persist_directory atau Settings
    vector_db = Chroma.from_documents(documents=docs, embedding=embeddings)
    return vector_db
