# app.py
import os
from io import BytesIO
from typing import List

import streamlit as st
from dotenv import load_dotenv

# File parsing
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation as PptxPresentation
from PIL import Image
import fitz  # PyMuPDF

# LangChain / VectorStore / Embeddings / LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# Optional OCR (pytesseract). Works only if tesseract binary installed.
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# -------------------------
# Config
# -------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
st.set_page_config(page_title="Gemini Multi-file Chatbot (with images & OCR)", page_icon="🤖", layout="wide")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY tidak ditemukan. Tambahkan di .env dulu.")
    st.stop()

# Embeddings and splitter
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

# -------------------------
# OCR helper (best-effort)
# -------------------------
def ocr_image_pytesseract(pil_image: Image.Image) -> str:
    """Gunakan pytesseract jika tersedia."""
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        text = pytesseract.image_to_string(pil_image, lang='eng')  # optional: change lang
        return text
    except Exception:
        return ""

# NOTE: If you want to use Google Vision API instead of pytesseract,
# implement a function ocr_image_with_google_vision(image_bytes) here.

# -------------------------
# Extractors per filetype
# -------------------------
def extract_text_from_pdf_bytes(raw_bytes: bytes) -> str:
    """1) Try extract text via PyPDF2. 2) If no text, render pages via PyMuPDF + OCR (if available)."""
    text = ""
    # Try PyPDF2 text extraction
    try:
        reader = PdfReader(BytesIO(raw_bytes))
        for page in reader.pages:
            ptext = page.extract_text()
            if ptext:
                text += ptext + "\n"
    except Exception:
        pass

    # If no text found and OCR is available, render pages and OCR them
    if (not text.strip()) and TESSERACT_AVAILABLE:
        try:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = ocr_image_pytesseract(img)
                if ocr_text:
                    text += ocr_text + "\n"
        except Exception:
            # best-effort: ignore if rendering fails
            pass
    return text

def extract_text_from_docx_bytes(raw_bytes: bytes) -> str:
    text = ""
    try:
        bio = BytesIO(raw_bytes)
        doc = DocxDocument(bio)
        for p in doc.paragraphs:
            if p.text:
                text += p.text + "\n"
    except Exception:
        pass
    return text

def extract_text_from_pptx_bytes(raw_bytes: bytes) -> str:
    text = ""
    try:
        bio = BytesIO(raw_bytes)
        prs = PptxPresentation(bio)
        # Extract textual content
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text += shape.text + "\n"
        # Also attempt to extract images from slides and OCR them if available
        if TESSERACT_AVAILABLE:
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "image") and shape.image:
                        img_bytes = shape.image.blob
                        pil = Image.open(BytesIO(img_bytes)).convert("RGB")
                        text += ocr_image_pytesseract(pil) + "\n"
    except Exception:
        pass
    return text

def extract_text_from_txt_bytes(raw_bytes: bytes) -> str:
    try:
        return raw_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def extract_text_from_image_bytes(raw_bytes: bytes) -> str:
    """Direct OCR from an uploaded image file"""
    try:
        pil = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        return ""
    # If pytesseract available, use it. Else return "" (no OCR).
    if TESSERACT_AVAILABLE:
        return ocr_image_pytesseract(pil)
    return ""

# -------------------------
# Build Documents and VectorStore
# -------------------------
def build_documents_from_uploads(uploaded_files) -> List[Document]:
    docs: List[Document] = []
    for f in uploaded_files:
        name = f.name
        raw = f.getvalue() if hasattr(f, "getvalue") else f.read()
        text = ""
        lower = name.lower()
        if lower.endswith(".pdf"):
            text = extract_text_from_pdf_bytes(raw)
        elif lower.endswith(".docx"):
            text = extract_text_from_docx_bytes(raw)
        elif lower.endswith(".pptx"):
            text = extract_text_from_pptx_bytes(raw)
        elif lower.endswith(".txt"):
            text = extract_text_from_txt_bytes(raw)
        elif lower.endswith((".png", ".jpg", ".jpeg", ".jfif", ".gif", ".bmp", ".webp", ".tiff")):
            text = extract_text_from_image_bytes(raw)
        elif lower.endswith(".doc") or lower.endswith(".ppt"):
            # legacy binary formats — warn user to convert
            st.warning(f"⚠️ File `{name}` berformat lama (.doc/.ppt). Silakan konversi ke .docx/.pptx jika ekstraksi kosong.")
            text = ""
        else:
            st.warning(f"⚠️ Tipe file `{name}` tidak didukung.")
            text = ""

        if not text or not text.strip():
            # skip empty extraction
            continue

        chunks = SPLITTER.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"source_file": name, "chunk_id": i}))
    return docs

def build_faiss_from_documents(docs: List[Document]) -> FAISS | None:
    if not docs:
        return None
    vs = FAISS.from_documents(docs, embedding=EMBEDDINGS)
    return vs

# -------------------------
# Prompt helpers
# -------------------------
def format_context(snippets: List[Document]) -> str:
    parts = []
    for idx, d in enumerate(snippets, start=1):
        src = d.metadata.get("source_file", "unknown")
        cid = d.metadata.get("chunk_id", "-")
        parts.append(f"[{idx}] ({src}#chunk-{cid})\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

def render_sources(snippets: List[Document]):
    with st.expander("🔎 Sumber konteks yang dipakai"):
        for i, d in enumerate(snippets, start=1):
            src = d.metadata.get("source_file", "unknown")
            cid = d.metadata.get("chunk_id", "-")
            preview = d.page_content[:300].replace("\n", " ")
            st.markdown(f"**[{i}]** **{src}** (chunk {cid})")
            st.caption(preview + ("..." if len(d.page_content) > 300 else ""))

# -------------------------
# UI: Sidebar (Upload & actions)
# -------------------------
st.sidebar.header("📂 Upload files (multi)")
uploaded_files = st.sidebar.file_uploader(
    "Upload multiple files: PDF, TXT, DOCX, PPTX, images (jpg/png/gif...), etc.",
    type=["pdf", "txt", "docx", "pptx", "png", "jpg", "jpeg", "jfif", "gif", "bmp", "webp", "tiff"],
    accept_multiple_files=True
)
build_btn = st.sidebar.button("🚀 Build Vector Store")
clear_btn = st.sidebar.button("🧹 Reset vector store")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if clear_btn:
    st.session_state.vector_store = None
    st.session_state.indexed_files = []
    st.success("Vector store di-reset.")

if build_btn:
    if not uploaded_files:
        st.sidebar.warning("Silakan upload minimal 1 file terlebih dahulu.")
    else:
        with st.spinner("📦 Memproses file dan membuat vector store..."):
            docs = build_documents_from_uploads(uploaded_files)
            if not docs:
                st.sidebar.error("Tidak ada teks valid yang berhasil diekstrak. Periksa file atau aktifkan OCR (lihat catatan).")
            else:
                vs = build_faiss_from_documents(docs)
                st.session_state.vector_store = vs
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.sidebar.success(f"Vector store terbangun. Dokumen terindeks: {len(st.session_state.indexed_files)} | Chunk total: {len(docs)}")

# -------------------------
# Main: Chat UI
# -------------------------
st.title("🤖 Gemini 2.5 Flash Chatbot — Multi-file (PDF/TXT/DOCX/PPTX/Images) ")
if TESSERACT_AVAILABLE:
    st.caption("OCR: pytesseract detected (Tesseract binary available). Images and PDFs will be OCR'd when needed.")
else:
    st.caption("OCR: pytesseract/Tesseract NOT available. Images will NOT be OCR'd. To enable OCR, install Tesseract on server or integrate external OCR API.")

if st.session_state.indexed_files:
    st.write("**Dokumen terindeks:**")
    st.write(" • " + "\n • ".join(st.session_state.indexed_files))

prompt = st.text_input("Tanyakan sesuatu berdasarkan dokumen yang diupload:", placeholder="Misal: Ringkas semua referensi tentang topik X...")
ask_btn = st.button("Tanyakan")

if ask_btn:
    if not prompt or not prompt.strip():
        st.warning("Masukkan pertanyaan terlebih dahulu.")
    elif st.session_state.vector_store is None:
        st.info("Belum ada vector store. Upload file dan klik 'Build Vector Store' di sidebar.")
    else:
        with st.spinner("🔎 Mengambil konteks dari vector store..."):
            results = st.session_state.vector_store.similarity_search(prompt, k=5)

        context_text = format_context(results)
        system_instructions = (
            "Jawablah pertanyaan pengguna seakurat mungkin dengan mengacu pada konteks di bawah. "
            "Jika jawaban tidak terdapat pada konteks, katakan: \"Jawaban tidak tersedia dalam konteks yang diberikan, tetapi pelajari dan berikan informasi dan link website sebagai sumber atau referensi tambahan untuk memperkuat insightful yang mendukung konteks atau topik yang ditanyakan.\" "
            "Berikan referensi [angka] dan nomor halamannya ke potongan konteks atau topik bila relevan."
        )

        composed_prompt = (
            f"{system_instructions}\n\n"
            f"=== KONTEX ===\n{context_text}\n\n"
            f"=== PERTANYAAN ===\n{prompt}\n\n"
            f"=== JAWABAN ==="
            f"=== REFERENSI/SUMBER ==="
        )

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        try:
            with st.spinner("🤖 Gemini sedang menjawab..."):
                response = llm.invoke(composed_prompt)
            st.subheader("💬 Jawaban")
            out_text = getattr(response, "content", None) or (response.candidates[0].content if getattr(response, "candidates", None) else str(response))
            st.write(out_text)
            render_sources(results)
        except Exception as e:
            st.error(f"❌ Error saat memanggil Gemini: {e}")

