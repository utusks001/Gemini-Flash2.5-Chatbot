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

# pdf images
from pdf2image import convert_from_bytes

# OCR pakai EasyOCR
import easyocr

# langchain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# LLM wrappers
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GOOGLE_G = True
except Exception:
    HAS_GOOGLE_G = False

try:
    from langchain.llms import LlamaCpp
    HAS_LLAMA = True
except Exception:
    HAS_LLAMA = False

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH")

st.set_page_config(page_title="Gemini / LLaMA OCR Chatbot", layout="wide", page_icon="🤖")

# ---------------------------------------------------------------------
# Embeddings & splitter
# ---------------------------------------------------------------------
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

# ---------------------------------------------------------------------
# OCR via EasyOCR
# ---------------------------------------------------------------------
@st.cache_resource
def get_easyocr_reader(lang="en"):
    langs = []
    if "eng" in lang: langs.append("en")
    if "ind" in lang: langs.append("id")
    if not langs: langs = ["en"]
    return easyocr.Reader(langs, gpu=False)

def ocr_image_easyocr(image: Image.Image, lang="eng") -> str:
    reader = get_easyocr_reader(lang)
    img_array = image.convert("RGB")
    results = reader.readtext(np.array(img_array))
    text = "\n".join([res[1] for res in results])
    return text

# ---------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes, ocr_lang="eng") -> str:
    text = ""
    try:
        reader = PdfReader(BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.warning(f"PDF text extraction warning: {e}")

    if not text.strip():
        try:
            images = convert_from_bytes(file_bytes, dpi=200)
            for img in images:
                t = ocr_image_easyocr(img, lang=ocr_lang)
                text += t + "\n"
        except Exception as e:
            st.warning(f"PDF OCR failed: {e}")
    return text

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore")

def extract_text_from_docx(file_bytes: bytes) -> str:
    text = ""
    try:
        doc = DocxDocument(BytesIO(file_bytes))
        for p in doc.paragraphs:
            if p.text:
                text += p.text + "\n"
    except Exception as e:
        st.warning(f"docx extraction error: {e}")
    return text

def extract_text_from_pptx(file_bytes: bytes) -> str:
    text = ""
    try:
        prs = PptxPresentation(BytesIO(file_bytes))
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text += shape.text + "\n"
    except Exception as e:
        st.warning(f"pptx extraction error: {e}")
    return text

def extract_text_from_image_bytes(file_bytes: bytes, lang="eng") -> str:
    try:
        img = Image.open(BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        st.warning(f"open image failed: {e}")
        return ""
    return ocr_image_easyocr(img, lang=lang)

def extract_text_from_file(uploaded_file, ocr_lang="eng") -> str:
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(raw, ocr_lang)
    if name.endswith(".txt"):
        return extract_text_from_txt(raw)
    if name.endswith(".docx"):
        return extract_text_from_docx(raw)
    if name.endswith(".pptx"):
        return extract_text_from_pptx(raw)
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
        if not text.strip():
            continue
        chunks = SPLITTER.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"source": f.name, "chunk_id": i}))
    return docs

def build_faiss_from_documents(docs: List[Document]) -> Optional[FAISS]:
    if not docs:
        return None
    return FAISS.from_documents(docs, embedding=EMBEDDINGS)

# ---------------------------------------------------------------------
# LLMs
# ---------------------------------------------------------------------
def get_llm(choice="gemini"):
    if choice == "gemini":
        if not HAS_GOOGLE_G:
            raise RuntimeError("langchain_google_genai tidak tersedia.")
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    elif choice == "llama":
        if not HAS_LLAMA or not LLAMA_MODEL_PATH:
            raise RuntimeError("LLaMA belum terkonfigurasi.")
        return LlamaCpp(model_path=LLAMA_MODEL_PATH, n_ctx=2048, temperature=0.2)
    else:
        raise ValueError("Pilihan LLM tidak dikenal.")

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title("🤖 Gemini / LLaMA OCR Multi-file Chatbot (EasyOCR + FAISS)")

with st.sidebar:
    uploaded_files = st.file_uploader(
        "Upload multiple files",
        type=["pdf","txt","docx","pptx","png","jpg","jpeg","bmp","jfif","gif"],
        accept_multiple_files=True
    )
    ocr_lang = st.selectbox("Bahasa OCR", ["eng","ind","eng+ind"], index=0)
    llm_choice = st.selectbox("Pilih LLM", ["gemini","llama"])
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
            vs = build_faiss_from_documents(docs)
            st.session_state.vector_store = vs
        st.success("Vector store terbangun.")
    else:
        st.warning("Upload file dulu.")

query = st.text_input("Pertanyaan:")
if st.button("Tanyakan"):
    if not query:
        st.warning("Isi pertanyaan.")
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
