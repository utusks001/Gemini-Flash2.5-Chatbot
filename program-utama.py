# Streamlit UI
# program-utama.py
"""
Gemini / Groq Image-enabled Multi-File Chatbot
================================================
Streamlit app yang bisa menerima file PDF, DOCX, PPTX, TXT **dan** gambar,
ekstraksi teks (OCR utk gambar), buat FAISS vector store, lalu jawab pertanyaan
pakai Gemini atau Groq Llama-3.1-405B-Instruct.

Author: OpenAI-ChatGPT
License: MIT
"""

import os
import io
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import easyocr
import numpy as np

# ---------------------------------
# LangChain imports
# ---------------------------------
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LLM wrappers
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

# ---------------------------------
# Environment / config
# ---------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not (GOOGLE_API_KEY or GROQ_API_KEY):
    st.error(
        "❌ **No LLM credentials found**.\n"
        "Set `GOOGLE_API_KEY` atau `GROQ_API_KEY` di `.env` atau Streamlit secrets."
    )
    st.stop()

# ---------------------------------
# Global objects
# ---------------------------------
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

# OCR reader (load sekali)
OCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)

# ---------------------------------
# File parsing helpers
# ---------------------------------
def extract_text_from_pdf(file_bytes):
    from pypdf import PdfReader
    text = ""
    try:
        reader = PdfReader(file_bytes)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.warning(f"⚠️ PDF extraction error: {e}")
    return text


def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.warning(f"⚠️ TXT read error: {e}")
        return ""


def extract_text_from_docx(file_bytes):
    from docx import Document as DocxDocument
    text = ""
    try:
        file_bytes.seek(0)
        doc = DocxDocument(file_bytes)
        for p in doc.paragraphs:
            if p.text:
                text += p.text + "\n"
    except Exception as e:
        st.warning(f"⚠️ DOCX extraction error: {e}")
    return text


def extract_text_from_pptx(file_bytes):
    from pptx import Presentation
    text = ""
    try:
        file_bytes.seek(0)
        prs = Presentation(file_bytes)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text += shape.text + "\n"
    except Exception as e:
        st.warning(f"⚠️ PPTX extraction error: {e}")
    return text


def extract_text_from_image(file_bytes):
    try:
        image = Image.open(file_bytes).convert("RGB")
        img_np = np.array(image)
        results = OCR_READER.readtext(img_np, detail=0)
        return "\n".join(results)
    except Exception as e:
        st.warning(f"⚠️ Image OCR error: {e}")
        return ""


def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()
    bio = io.BytesIO(raw_bytes)

    if name.endswith(".pdf"):
        return extract_text_from_pdf(bio)
    elif name.endswith(".txt"):
        return extract_text_from_txt(bio)
    elif name.endswith(".docx"):
        return extract_text_from_docx(bio)
    elif name.endswith(".pptx"):
        return extract_text_from_pptx(bio)
    elif name.endswith((".bmp", ".png", ".jpg", ".jpeg", ".gif", ".jfif")):
        return extract_text_from_image(bio)
    else:
        st.warning(f"⚠️ Unsupported file type: `{uploaded_file.name}`")
        return ""


# ---------------------------------
# Build documents and vector store
# ---------------------------------
def build_documents_from_uploads(uploaded_files):
    docs = []
    for f in uploaded_files:
        text = extract_text_from_file(f)
        if not text or not text.strip():
            continue
        chunks = SPLITTER.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source_file": f.name, "chunk_id": i},
                )
            )
    return docs


def build_faiss_from_documents(docs):
    if not docs:
        return None
    return FAISS.from_documents(docs, embedding=EMBEDDINGS)


# ---------------------------------
# Prompt helpers
# ---------------------------------
def format_context(snippets):
    parts = []
    for idx, d in enumerate(snippets, start=1):
        src = d.metadata.get("source_file", "unknown")
        cid = d.metadata.get("chunk_id", "-")
        parts.append(f"[{idx}] ({src}#chunk-{cid})\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def render_sources(snippets):
    with st.expander("🔎 Sumber konteks yang dipakai"):
        for i, d in enumerate(snippets, start=1):
            src = d.metadata.get("source_file", "unknown")
            cid = d.metadata.get("chunk_id", "-")
            preview = d.page_content[:300].replace("\n", " ")
            st.markdown(f"**[{i}]** **{src}** (chunk {cid})")
            st.caption(preview + ("..." if len(d.page_content) > 300 else ""))


# ---------------------------------
# LLM helper
# ---------------------------------
def get_llm():
    if GOOGLE_API_KEY and ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=GOOGLE_API_KEY,
        )
    elif GROQ_API_KEY and ChatGroq:
        return ChatGroq(
            model="llama-3.1-405b-instruct",
            temperature=0.2,
            groq_api_key=GROQ_API_KEY,
        )
    else:
        st.error("❌ LLM library not available.")
        return None


# ---------------------------------
# Streamlit UI
# ---------------------------------
st.set_page_config(
    page_title="Gemini / Groq Image-Chatbot – Multi-File, OCR, FAISS",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Gemini / Groq Image-Chatbot – Multi-File, OCR, FAISS")
st.markdown(
    """
Upload file **PDF, TXT, DOCX, PPTX, BMP, PNG, JPG, JPEG, GIF, JFIF**.  
Pipeline:
1. **Extract** teks (OCR utk gambar)  
2. **Chunk** teks  
3. **Embed** & index FAISS  
4. **Answer** pertanyaan via Gemini / Groq
"""
)

# Sidebar – upload
st.sidebar.header("📂 Upload & Build")
uploaded_files = st.sidebar.file_uploader(
    "Upload files",
    type=["pdf", "txt", "docx", "pptx", "bmp", "png", "jpg", "jpeg", "gif", "jfif"],
    accept_multiple_files=True,
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
    st.sidebar.success("Vector store di-reset.")

if build_btn:
    if not uploaded_files:
        st.sidebar.warning("Silakan upload minimal 1 file dahulu.")
    else:
        with st.spinner("📦 Memproses file & membuat vector store..."):
            docs = build_documents_from_uploads(uploaded_files)
            if not docs:
                st.sidebar.error("Tidak ada teks valid dari file.")
            else:
                vs = build_faiss_from_documents(docs)
                st.session_state.vector_store = vs
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.sidebar.success(
                    f"Vector store terbangun ✅ Dokumen: {len(st.session_state.indexed_files)} | Chunk: {len(docs)}"
                )

if st.session_state.indexed_files:
    st.markdown("**Dokumen terindeks:**")
    st.write(" • " + "\n • ".join(st.session_state.indexed_files))

# Query
st.subheader("🗣️ Tanya sesuatu...")
prompt = st.text_input("Pertanyaan:", placeholder="Contoh: Ringkas isi dokumen...")
ask_btn = st.button("Tanyakan")

if ask_btn:
    if not prompt.strip():
        st.warning("Masukkan pertanyaan terlebih dahulu.")
    elif st.session_state.vector_store is None:
        st.info("Belum ada vector store. Upload file + Build Vector Store dulu.")
    else:
        with st.spinner("🔎 Mencari konteks…"):
            results = st.session_state.vector_store.similarity_search(prompt, k=5)

        context_text = format_context(results)
        system_instructions = (
            "Jawablah seakurat dan sedetil mungkin berdasarkan konteks berikut. "
            "Jika jawaban tidak ada, katakan: "
            "\"Jawaban tidak tersedia dalam konteks yang diberikan, "
            "tambahkan informasi dari internet dengan link.\" "
            "Sertakan referensi [angka] jika relevan."
        )

        composed_prompt = (
            f"{system_instructions}\n\n"
            f"=== KONTEX ===\n{context_text}\n\n"
            f"=== PERTANYAAN ===\n{prompt}\n\n"
            f"=== JAWABAN ==="
        )

        llm = get_llm()
        if llm is None:
            st.error("❌ LLM tidak bisa dipanggil.")
        else:
            try:
                with st.spinner("🤖 Menjawab…"):
                    response = llm.invoke(composed_prompt)

                st.subheader("💬 Jawaban")
                out_text = getattr(response, "content", None) or str(response)
                st.write(out_text)

                render_sources(results)

            except Exception as e:
                st.error(f"❌ Error saat memanggil LLM: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
**Disclaimer**  
OCR bergantung pada resolusi gambar.  
Jawaban dihasilkan LLM, bukan pengganti saran profesional.
"""
)
