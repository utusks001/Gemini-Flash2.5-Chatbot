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
import requests
import docx
import pptx
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# === Load API Keys from secrets ===
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
OCRSPACE_API_KEY = st.secrets.get("OCRSPACE_API_KEY", "")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# === Utility: extract text from files ===
def read_pdf(file):
    text = ""
    pdf = PdfReader(file)
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pptx(file):
    prs = pptx.Presentation(file)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

def read_txt(file):
    return file.read().decode("utf-8")

def read_image(file):
    url_api = "https://api.ocr.space/parse/image"
    payload = {
        "isOverlayRequired": False,
        "apikey": OCRSPACE_API_KEY,
        "language": "eng"
    }
    files = {"file": (file.name, file, file.type)}
    response = requests.post(url_api, files=files, data=payload)
    result = response.json()
    if result.get("IsErroredOnProcessing"):
        return "[OCR Error] " + str(result.get("ErrorMessage"))
    text = ""
    for parsed in result.get("ParsedResults", []):
        text += parsed.get("ParsedText", "")
    return text

# === Build documents from uploaded files ===
def build_documents_from_uploads(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name.lower()
        text = ""

        if filename.endswith(".pdf"):
            text = read_pdf(uploaded_file)
        elif filename.endswith(".docx"):
            text = read_docx(uploaded_file)
        elif filename.endswith(".pptx"):
            text = read_pptx(uploaded_file)
        elif filename.endswith(".txt"):
            text = read_txt(uploaded_file)
        elif filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".jfif")):
            text = read_image(uploaded_file)
        else:
            text = "Unsupported file type."

        if text.strip():
            documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents

# === Main App ===
def main():
    st.title("📚 Multi-file RAG Chatbot (OCR.space + Gemini/Groq)")

    with st.sidebar:
        st.subheader("⚙️ Settings")
        llm_choice = st.radio(
            "Select LLM Provider:",
            options=["Gemini", "Groq"] if GROQ_API_KEY else ["Gemini"]
        )
        if llm_choice == "Groq":
            model_name = st.selectbox(
                "Groq Model:",
                ["llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma-7b-it"]
            )
        else:
            model_name = st.selectbox(
                "Gemini Model:",
                ["gemini-1.5-flash", "gemini-1.5-pro"]
            )

    uploaded_files = st.file_uploader(
        "Upload multiple files (pdf, docx, pptx, txt, images)...",
        type=["pdf", "docx", "pptx", "txt", "png", "jpg", "jpeg", "bmp", "gif", "jfif"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

        if st.button("Build Knowledge Base"):
            with st.spinner("Processing files and building vector store..."):
                documents = build_documents_from_uploads(uploaded_files)
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = splitter.split_documents(documents)

                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vectorstore = FAISS.from_documents(texts, embeddings)
                st.session_state.vs = vectorstore
            st.success("Knowledge base built!")

    query = st.text_input("Ask a question about your documents:")

    if query and "vs" in st.session_state:
        docs = st.session_state.vs.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        if llm_choice == "Gemini":
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
            response = llm.predict(f"Answer the question based on context:\n\n{context}\n\nQuestion: {query}")
        else:
            llm = ChatGroq(model=model_name, temperature=0)
            response = llm.predict(f"Answer the question based on context:\n\n{context}\n\nQuestion: {query}")

        st.subheader("Answer:")
        st.write(response)

        with st.expander("Retrieved context"):
            st.write(context)

if __name__ == "__main__":
    main()
