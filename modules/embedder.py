# Embedding & vector store
# modules/embedder.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def build_chroma_from_documents(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(docs, embedding=embeddings)  # No persist_directory
    return vector_db
