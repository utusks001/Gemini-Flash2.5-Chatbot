import os
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv
from typing import List

# File parsing
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation as PptxPresentation
import easyocr
import numpy as np

# LangChain / VectorStore / Embeddings / LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# -------------------------
# Config / env
# -------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="Multi-file Chatbot (Gemini/LLaMA + FAISS)",
    page_icon="🤖",
    layout="wide"
)

# Embeddings (HuggingFace yang stabil di Streamlit Cloud)
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Text splitter
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

# Inisialisasi EasyOCR Reader
# Jalankan sekali saat startup untuk efisiensi
@st.cache_resource
def get_ocr_reader():
    """Menginisialisasi EasyOCR Reader."""
    return easyocr.Reader(['en', 'id'])

reader = get_ocr_reader()

# -------------------------
# File extractors (tambahan untuk images)
# -------------------------
def extract_text_from_pdf(file_bytes: BytesIO) -> str:
    text = ""
    try:
        reader = PdfReader(file_bytes)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.warning(f"⚠️ Gagal ekstrak PDF: {e}")
    return text


def extract_text_from_txt(file_bytes: BytesIO) -> str:
    try:
        return file_bytes.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.warning(f"⚠️ Gagal baca TXT: {e}")
        return ""


def extract_text_from_docx(file_bytes: BytesIO) -> str:
    text = ""
    try:
        file_bytes.seek(0)
        doc = DocxDocument(file_bytes)
        for p in doc.paragraphs:
            if p.text:
                text += p.text + "\n"
    except Exception as e:
        st.warning(f"⚠️ Gagal ekstrak DOCX: {e}")
    return text


def extract_text_from_pptx(file_bytes: BytesIO) -> str:
    text = ""
    try:
        file_bytes.seek(0)
        prs = PptxPresentation(file_bytes)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text += shape.text + "\n"
    except Exception as e:
        st.warning(f"⚠️ Gagal ekstrak PPTX: {e}")
    return text

def extract_text_from_image(file_bytes: BytesIO) -> str:
    text = ""
    try:
        img_array = np.frombuffer(file_bytes.getvalue(), np.uint8)
        results = reader.readtext(img_array)
        for (bbox, text_content, prob) in results:
            text += text_content + "\n"
    except Exception as e:
        st.warning(f"⚠️ Gagal ekstrak teks dari gambar: {e}")
    return text

def extract_text_from_file(uploaded_file) -> str:
    """Best-effort generic extractor"""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    bio = BytesIO(raw)

    if name.endswith(".pdf"):
        return extract_text_from_pdf(bio)
    elif name.endswith(".txt"):
        return extract_text_from_txt(BytesIO(raw))
    elif name.endswith(".docx"):
        return extract_text_from_docx(BytesIO(raw))
    elif name.endswith(".pptx"):
        return extract_text_from_pptx(BytesIO(raw))
    elif name.endswith((".bmp", ".png", ".jfif", ".jpg", ".jpeg", ".gif")):
        return extract_text_from_image(bio)
    elif name.endswith((".doc", ".ppt")):
        st.warning(f"⚠️ File `{uploaded_file.name}` berformat lama (.doc/.ppt). Silakan konversi ke .docx/.pptx untuk ekstraksi teks yang lebih baik.")
        return ""
    else:
        st.warning(f"⚠️ Tipe file `{uploaded_file.name}` tidak didukung.")
        return ""

# -------------------------
# Build documents & FAISS
# -------------------------
def build_documents_from_uploads(uploaded_files) -> List[Document]:
    docs: List[Document] = []
    for f in uploaded_files:
        text = extract_text_from_file(f)
        if not text or not text.strip():
            continue
        chunks = SPLITTER.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"source_file": f.name, "chunk_id": i}))
    return docs


def build_faiss_from_documents(docs: List[Document]) -> FAISS | None:
    if not docs:
        return None
    vs = FAISS.from_documents(docs, embedding=EMBEDDINGS)
    return vs

# -------------------------
# Prompt formatting helpers
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
# Streamlit UI
# -------------------------
st.title("🤖 Multi-file Chatbot (Gemini / LLaMA) + RAG")
st.write("Upload banyak file (PDF, TXT, DOCX, PPTX, dan Gambar).")

# Sidebar
st.sidebar.header("📂 Upload & Build")
uploaded_files = st.sidebar.file_uploader(
    "Upload files (pdf, txt, docx, pptx, jpg, png, dll.) — boleh banyak",
    type=["pdf", "txt", "docx", "pptx", "jpg", "jpeg", "png", "gif", "bmp", "jfif"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Konfigurasi LLM")
llm_options = []
if GOOGLE_API_KEY:
    llm_options.append("Gemini (Google)")
if GROQ_API_KEY:
    llm_options.append("LLaMA (GROQ)")

selected_llm = None
if llm_options:
    selected_llm = st.sidebar.radio("Pilih LLM:", llm_options)
else:
    st.sidebar.error("❌ Tidak ada API Key yang ditemukan.")
    st.sidebar.stop()

st.sidebar.markdown("---")
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
                st.sidebar.error("Tidak ada teks valid yang berhasil diekstrak. Periksa file atau konversi ke format yang didukung.")
            else:
                vs = build_faiss_from_documents(docs)
                st.session_state.vector_store = vs
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.sidebar.success(f"Vector store terbangun. Dokumen terindeks: {len(st.session_state.indexed_files)} | Chunk total: {len(docs)}")

# Show indexed files
if st.session_state.indexed_files:
    st.markdown("**Dokumen terindeks:**")
    st.write(" • " + "\n • ".join(st.session_state.indexed_files))

# Query area
prompt = st.text_input(
    "Tanyakan sesuatu berdasarkan dokumen yang diupload:",
    placeholder="Misal: Ringkas dokumen tentang topik X..."
)
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
            "Jawablah seakurat dan sedetil mungkin sehingga mudah dipahami berdasarkan konteks berikut. "
            "Jika jawaban tidak ada, katakan: "
            "\"Jawaban tidak tersedia dalam konteks yang diberikan, "
            "pelajari dan berikan tambahan informasi yang diperlukan sesuai konteks atau topik  "
            "yang ditanyakan dari berbagai sumber atau referensi di internet dengan menyertakan link website.\" "
            "Sertakan referensi [angka] ke potongan konteks bila relevan."
        )

        # Inisialisasi LLM berdasarkan pilihan pengguna
        llm = None
        if selected_llm == "Gemini (Google)":
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        elif selected_llm == "LLaMA (GROQ)":
            llm = ChatGroq(model="llama3-8b-8192", temperature=0.2, api_key=GROQ_API_KEY)

        if not llm:
            st.error("Gagal menginisialisasi LLM. Periksa konfigurasi API Key Anda.")
        else:
            try:
                with st.spinner(f"🤖 {selected_llm} sedang menjawab..."):
                    document_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_messages([
                        ("system", system_instructions + "\n\nKonteks:\n{context}"),
                        ("human", "{input}"),
                    ]))
                    retrieval_chain = create_retrieval_chain(st.session_state.vector_store.as_retriever(), document_chain)
                    response = retrieval_chain.invoke({"input": prompt})

                st.subheader("💬 Jawaban")
                st.write(response["answer"])
                render_sources(results)

            except Exception as e:
                st.error(f"❌ Error saat memanggil LLM: {e}")
