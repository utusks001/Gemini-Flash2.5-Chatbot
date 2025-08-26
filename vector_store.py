# vector_store.py
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Force Chroma to run in client-only mode
    chroma_settings = Settings(anonymized_telemetry=False, allow_reset=True, is_persistent=False)

    vector_store = Chroma.from_texts(
        chunks,
        embedding=embeddings,
        client_settings=chroma_settings
    )
    return vector_store










