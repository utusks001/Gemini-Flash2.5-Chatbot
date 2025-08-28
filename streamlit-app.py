# app.py
# app.py
import os
from io import BytesIO
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

# file parsing
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation as PptxPresentation
from PIL import Image
from pdf2image import convert_from_bytes

# OCR pakai EasyOCR
import numpy as np
import easyocr

# langchain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# LLM clients
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Gemini + Groq OCR Chatbot", layout="wide", page_icon="🤖")

# ---------------------------------------------------------------------
# Embeddings & splitter
# ---------------------------------------------------------------------
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

# ---------------------------------------------------------------------
# OCR via EasyOCR
# ---------------------------------------------------------------------
@st.cache_resource
def get_easyocr_reader(lang="eng"):
    langs = []
    if "eng" in lang: langs.append("en")
    if "ind" in lang: langs.append("id")
    if not langs: langs = ["en"]
    return easyocr.Reader(langs, gpu=False)

def ocr_image_easyocr(image: Image.Image, lang="eng") -> str:
    reader = get_easyocr_reader(lang)
    results = reader.readtext(np.array(image))
    return "\n".join([res[1] for res in results])

# ---------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes, ocr_lang="eng") -> str:
    text = ""
    try:
        reader = PdfReader(BytesIO(file_bytes))
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    except Exception:
        pass

    if not text.strip():
        try:
            images = convert_from_bytes(file_bytes, dpi=200)
            for img in images:
                text += ocr_image_easyocr(img, lang=ocr_lang) + "\n"
        except Exception as e:
            st.warning(f"PDF OCR failed: {e}")
    return text

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore")

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text)

def extract_text_from_pptx(file_bytes: bytes) -> str:
    prs = PptxPresentation(BytesIO(file_bytes))
    texts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                texts.append(shape.text)
    return "\n".join(texts)

def extract_text_from_image_bytes(file_bytes: bytes, lang="eng") -> str:
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    return ocr_image_easyocr(img, lang=lang)

def extract_text_from_file(uploaded_file, ocr_lang="eng") -> str:
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    if name.endswith(".pdf"): return extract_text_from_pdf(raw, ocr_lang)
    if name.endswith(".txt"): return extract_text_from_txt(raw)
    if name.endswith(".docx"): return extract_text_from_docx(raw)
    if name.endswith(".pptx"): return extract_text_from_pptx(raw)
    if name.endswith((".png",".jpg",".jpeg",".bmp",".jfif",".gif")):
        return extract_text_from_image_bytes(raw, lang=ocr_lang)
    st.warning(f"Tipe file {uploaded_file.name} tidak didukung.")
    return ""

# ---------------------------------------------------------------------
# Build documents & FAISS
# ---------------------------------------------------------------------
def build_documents_from_uploads(uploaded_files, ocr_lang="eng") -> List[Document]:
    docs = []
    for f in uploaded_files:
        text = extract_text_from_file(f, ocr_lang=ocr_lang)
        if text.strip():
            chunks = SPLITTER.split_text(text)
            for i, chunk in enumerate(chunks):
                docs.append(Document(page_content=chunk, metadata={"source": f.name, "chunk": i}))
    return docs

def build_faiss_from_documents(docs: List[Document]) -> Optional[FAISS]:
    if not docs: return None
    return FAISS.from_documents(docs, embedding=EMBEDDINGS)

# ---------------------------------------------------------------------
# LLMs
# ---------------------------------------------------------------------
def get_llm(choice="gemini"):
    if choice == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    elif choice == "groq":
        return ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192", temperature=0.2)
    else:
        raise ValueError("Pilih gemini atau groq.")

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title("🤖 Gemini + Groq OCR Multi-file Chatbot (EasyOCR + FAISS)")

with st.sidebar:
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf","txt","docx","pptx","png","jpg","jpeg","bmp","jfif","gif"],
        accept_multiple_files=True
    )
    ocr_lang = st.selectbox("Bahasa OCR", ["eng","ind","eng+ind"], index=0)
    llm_choice = st.selectbox("Pilih LLM", ["gemini","groq"])
    build_btn = st.button("🚀 Build Vector Store")
    clear_btn = st.button("🧹 Reset")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if clear_btn:
    st.session_state.vector_store = None
    st.success("Vector store direset.")

if build_btn:
    if uploaded_files:
        with st.spinner("Membangun vector store..."):
            docs = build_documents_from_uploads(uploaded_files, ocr_lang=ocr_lang)
            st.session_state.vector_store = build_faiss_from_documents(docs)
        st.success("Vector store terbangun.")
    else:
        st.warning("Upload file dulu.")

query = st.text_input("Pertanyaan:")
if st.button("Tanyakan"):
    if not query:
        st.warning("Isi pertanyaan dulu.")
    elif not st.session_state.vector_store:
        st.warning("Belum ada vector store.")
    else:
        retriever = st.session_state.vector_store.as_retriever()
        llm = get_llm(llm_choice)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        with st.spinner("Menjawab..."):
            ans = qa.run(query)
        st.write("### 💬 Jawaban")
        st.write(ans)
