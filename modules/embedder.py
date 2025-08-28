# Embedding & vector store
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def build_chroma_from_documents(docs, persist_dir="chroma_store"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    return vector_db
