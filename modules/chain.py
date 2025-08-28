# Integrasi LLM + retriever
from langchain.chains import RetrievalQA

def build_qa_chain(llm, retriever):
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
