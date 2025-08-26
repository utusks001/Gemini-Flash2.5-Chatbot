# text_splitter.py

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text, chunk_size=10000, chunk_overlap=1000):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)
