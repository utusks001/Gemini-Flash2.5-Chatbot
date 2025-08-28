# FAISS Builder
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vector_store(text_chunks):
    docs = [Document(page_content=chunk) for chunk in text_chunks]
    return FAISS.from_documents(docs, embedding=embedding_model)

def search_vector_store(store, query, k=5):
    return store.similarity_search(query, k=k)
