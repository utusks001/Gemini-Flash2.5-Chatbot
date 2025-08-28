# Embedding & vector store
# modules/embedder.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from chromadb.config import Settings

def build_chroma_from_documents(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # In-memory Chroma client settings
    chroma_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=None,
        anonymized_telemetry=False
    )

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        client_settings=chroma_settings
    )
    return vector_db
