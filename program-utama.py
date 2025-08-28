# Streamlit UI
# program-utama.py
"""
Gemini / Groq Image-enabled Multi-File Chatbot
================================================
Streamlit app yang bisa menerima file PDF, DOCX, PPTX, TXT **dan** gambar,
ekstraksi teks (OCR utk gambar), buat FAISS vector store, lalu jawab pertanyaan
pakai Gemini atau Groq 
"""
import streamlit as st
import os
import pytesseract
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
from PIL import Image
from io import BytesIO

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain


# Load API key dari Streamlit secrets
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# OCR untuk gambar dengan Tesseract
def extract_text_from_image(file):
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image, lang="eng+ind")
        return text
    except Exception as e:
        return f"OCR failed: {e}"


# Parsing berbagai jenis dokumen
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


# Membuat vectorstore dari dokumen
def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    return FAISS.from_documents(docs, embeddings)


# Aplikasi Streamlit
def main():
    st.title("📚 Multi-file Chatbot with OCR (Tesseract + Gemini)")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, PPTX, TXT, atau Images",
        type=["pdf", "docx", "pptx", "txt", "png", "jpg", "jpeg", "gif", "bmp", "jfif"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Memproses dokumen..."):
            documents = load_documents(uploaded_files)
            vectorstore = build_vectorstore(documents)
            retriever = vectorstore.as_retriever()
            chat_model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
            qa = ConversationalRetrievalChain.from_llm(chat_model, retriever)

        chat_history = []
        query = st.text_input("Tanyakan sesuatu dari dokumen Anda:")
        if query:
            result = qa({"question": query, "chat_history": chat_history})
            st.write("**Jawaban:**", result["answer"])
            chat_history.append((query, result["answer"]))


if __name__ == "__main__":
    main()

