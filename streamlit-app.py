# app.py
import streamlit as st
import os
import requests
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
from PIL import Image

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain

from langchain_groq import ChatGroq   # tambahan

# Load secrets
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OCRSPACE_API_KEY = os.getenv("OCRSPACE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# OCR with OCR.space API
def extract_text_from_image(file):
    try:
        url = "https://api.ocr.space/parse/image"
        payload = {"apikey": OCRSPACE_API_KEY, "language": "eng,ind"}
        files = {"file": file.getvalue()}
        response = requests.post(url, data=payload, files={"file": file})
        result = response.json()
        if result.get("ParsedResults"):
            return result["ParsedResults"][0]["ParsedText"]
        return ""
    except Exception as e:
        return f"OCR failed: {e}"

# Extract text from uploaded files
def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name.lower()

        if filename.endswith(".pdf"):
            pdf = PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            documents.append(Document(page_content=text, metadata={"source": filename}))

        elif filename.endswith(".docx"):
            doc = DocxDocument(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            documents.append(Document(page_content=text, metadata={"source": filename}))

        elif filename.endswith(".pptx"):
            prs = Presentation(uploaded_file)
            text_runs = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text_runs.append(shape.text)
            documents.append(Document(page_content="\n".join(text_runs), metadata={"source": filename}))

        elif filename.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".jfif")):
            text = extract_text_from_image(uploaded_file)
            documents.append(Document(page_content=text, metadata={"source": filename}))

        elif filename.endswith(".txt"):
            stringio = uploaded_file.getvalue().decode("utf-8")
            documents.append(Document(page_content=stringio, metadata={"source": filename}))

    return documents

# Build vector store
def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    return FAISS.from_documents(docs, embeddings)

# Main Streamlit app
def main():
    st.title("📚 Multi-file Chatbot with OCR (OCR.space + Gemini/Groq)")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, PPTX, TXT, or Images",
        type=["pdf", "docx", "pptx", "txt", "png", "jpg", "jpeg", "gif", "bmp", "jfif"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing documents..."):
            documents = load_documents(uploaded_files)
            vectorstore = build_vectorstore(documents)
            retriever = vectorstore.as_retriever()

            # pilih LLM
            llm_choice = st.radio("Pilih LLM:", ["Gemini (Google)", "Groq"])
            if llm_choice == "Gemini (Google)":
                chat_model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
            else:
                chat_model = ChatGroq(model="mixtral-8x7b-32768", groq_api_key=GROQ_API_KEY)

            qa = ConversationalRetrievalChain.from_llm(chat_model, retriever)

        chat_history = []
        query = st.text_input("Ask a question about your documents:")
        if query:
            result = qa({"question": query, "chat_history": chat_history})
            st.write("**Answer:**", result["answer"])
            chat_history.append((query, result["answer"]))

if __name__ == "__main__":
    main()
