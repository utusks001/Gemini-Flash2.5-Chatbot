# vector_store.py
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_texts(chunks, embedding=embeddings)
    return vector_store








