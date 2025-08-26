# app.py
import os
from io import BytesIO
from typing import List
import json

import streamlit as st
from PIL import Image
import fitz  # pymupdf
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation as PptxPresentation

# Google Vision client
from google.cloud import vision
from google.oauth2 import service_account

# LangChain / VectorStore / Embeddings / LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Config / defaults ---
st.set_page_config(page_title="Gemini Multi-file Chatbot Multi-file (PDF/TXT/DOCX/PPTX/Images)", page_icon="🤖", layout="wide")
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_MAX_FILES = 20
DEFAULT_MAX_CHUNKS = 2000
FAISS_PATH = "faiss_index"

# Embeddings (HuggingFace)
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Session state initialization
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# --- Utilities: load Vision client from Streamlit secrets (service account JSON) ---
def load_vision_client_from_secrets():
    """
    Expects st.secrets["gcp_service_account"] to exist with the service account JSON fields.
    Returns a google.cloud.vision.ImageAnnotatorClient
    """
    if "gcp_service_account" not in st.secrets:
        st.warning("Service account not found in Streamlit secrets (gcp_service_account). Vision OCR will be disabled.")
        return None
    try:
        creds_info = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_info)
        client = vision.ImageAnnotatorClient(credentials=creds)
        return client
    except Exception as e:
        st.error(f"Failed to create Vision client from secrets: {e}")
        return None

VISION_CLIENT = load_vision_client_from_secrets()

# --- OCR helpers ---
def ocr_image_via_vision_api(image_bytes: bytes) -> str:
    """Call google vision client DOCUMENT_TEXT_DETECTION"""
    if VISION_CLIENT is None:
        return ""
    try:
        image = vision.Image(content=image_bytes)
        response = VISION_CLIENT.document_text_detection(image=image)
        if response.error.message:
            st.warning(f"Vision OCR error: {response.error.message}")
            return ""
        annotation = response.full_text_annotation
        return annotation.text if annotation is not None else ""
    except Exception as e:
        st.warning(f"Vision OCR Exception: {e}")
        return ""

# --- File extractors ---
def extract_text_from_pdf_bytes(raw_bytes: bytes, use_ocr: bool = True) -> str:
    text = ""
    # Try PyPDF2 extraction
    try:
        reader = PdfReader(BytesIO(raw_bytes))
        for page in reader.pages:
            ptext = page.extract_text()
            if ptext:
                text += ptext + "\n"
    except Exception:
        pass

    # If empty and use_ocr, render pages and OCR
    if (not text.strip()) and use_ocr and VISION_CLIENT:
        try:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes()
                ocr_txt = ocr_image_via_vision_api(img_bytes)
                if ocr_txt:
                    text += ocr_txt + "\n"
        except Exception:
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

def extract_text_from_pptx_bytes(raw_bytes: bytes, use_ocr: bool = True) -> str:
    text = ""
    try:
        bio = BytesIO(raw_bytes)
        prs = PptxPresentation(bio)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text += shape.text + "\n"
        # OCR images in slides if any
        if use_ocr and VISION_CLIENT:
            for slide in prs.slides:
                for shape in slide.shapes:
                    try:
                        if hasattr(shape, "image") and shape.image:
                            img_bytes = shape.image.blob
                            ocr_txt = ocr_image_via_vision_api(img_bytes)
                            if ocr_txt:
                                text += ocr_txt + "\n"
                    except Exception:
                        continue
    except Exception:
        pass
    return text

def extract_text_from_txt_bytes(raw_bytes: bytes) -> str:
    try:
        return raw_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def extract_text_from_image_bytes(raw_bytes: bytes) -> str:
    if not VISION_CLIENT:
        return ""
    return ocr_image_via_vision_api(raw_bytes)

# --- Build documents & FAISS ---
def build_documents_from_uploads(uploaded_files, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP, max_files=DEFAULT_MAX_FILES, max_total_chunks=DEFAULT_MAX_CHUNKS) -> List[Document]:
    docs: List[Document] = []
    total_chunks_local = 0
    file_count = 0
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for f in uploaded_files:
        if file_count >= max_files:
            st.warning(f"Reached max files limit ({max_files}). Remaining files ignored.")
            break
        fname = f.name
        lower = fname.lower()
        raw = f.getvalue() if hasattr(f, "getvalue") else f.read()
        text = ""

        if lower.endswith(".pdf"):
            text = extract_text_from_pdf_bytes(raw, use_ocr=True)
        elif lower.endswith(".docx"):
            text = extract_text_from_docx_bytes(raw)
        elif lower.endswith(".pptx"):
            text = extract_text_from_pptx_bytes(raw, use_ocr=True)
        elif lower.endswith(".txt"):
            text = extract_text_from_txt_bytes(raw)
        elif lower.endswith((".png", ".jpg", ".jpeg", ".jfif", ".gif", ".bmp", ".webp", ".tiff")):
            text = extract_text_from_image_bytes(raw)
        elif lower.endswith(".doc") or lower.endswith(".ppt"):
            st.warning(f"File {fname} is legacy (.doc/.ppt). Convert to .docx/.pptx for better extraction.")
            text = ""
        else:
            st.warning(f"Unsupported file type: {fname}")
            text = ""

        if not text or not text.strip():
            file_count += 1
            continue

        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            if total_chunks_local >= max_total_chunks:
                st.warning(f"Reached max total chunks ({max_total_chunks}). Stopping chunking further files.")
                break
            docs.append(Document(page_content=chunk, metadata={"source_file": fname, "chunk_id": i}))
            total_chunks_local += 1

        file_count += 1
        if total_chunks_local >= max_total_chunks:
            break

    st.session_state.total_chunks = total_chunks_local
    return docs

def build_faiss_from_documents(docs: List[Document], save_local: bool = True, faiss_path: str = FAISS_PATH) -> FAISS | None:
    if not docs:
        return None
    vs = FAISS.from_documents(docs, embedding=EMBEDDINGS)
    if save_local:
        try:
            vs.save_local(faiss_path)
        except Exception as e:
            st.warning(f"Could not save FAISS to disk: {e}")
    return vs

def load_faiss_if_exists(faiss_path=FAISS_PATH):
    try:
        if os.path.exists(faiss_path):
            vs = FAISS.load_local(faiss_path, EMBEDDINGS, allow_dangerous_deserialization=True)
            return vs
    except Exception as e:
        st.warning(f"Failed to load local FAISS index: {e}")
    return None

# --- UI: Sidebar settings & upload ---
st.sidebar.header("Upload & Settings")
max_files = st.sidebar.number_input("Max files to index (per build)", min_value=1, max_value=50, value=DEFAULT_MAX_FILES)
max_total_chunks = st.sidebar.number_input("Max total chunks (per build)", min_value=100, max_value=50000, value=DEFAULT_MAX_CHUNKS, step=100)
chunk_size = st.sidebar.number_input("Chunk size (characters)", min_value=200, max_value=5000, value=DEFAULT_CHUNK_SIZE, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap (characters)", min_value=0, max_value=1000, value=DEFAULT_CHUNK_OVERLAP, step=10)

uploaded_files = st.sidebar.file_uploader("Upload files (pdf, txt, docx, pptx, images...)", type=["pdf", "txt", "docx", "pptx", "png", "jpg", "jpeg", "jfif", "gif", "bmp", "webp", "tiff"], accept_multiple_files=True)
col1, col2, col3 = st.sidebar.columns(3)
build_btn = col1.button("Build Vector Store")
load_btn = col2.button("Load saved FAISS")
clear_btn = col3.button("Clear vector store")
if clear_btn:
    st.session_state.vector_store = None
    st.session_state.indexed_files = []
    st.session_state.total_chunks = 0
    try:
        import shutil
        if os.path.exists(FAISS_PATH):
            shutil.rmtree(FAISS_PATH)
    except Exception:
        pass
    st.success("Cleared vector store (session + disk)")

if load_btn:
    loaded = load_faiss_if_exists(FAISS_PATH)
    if loaded:
        st.session_state.vector_store = loaded
        st.session_state.indexed_files = ["(loaded from disk)"]
        st.success("Loaded FAISS index from disk.")
    else:
        st.warning("No FAISS index found on disk.")

if build_btn:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one file to build the index.")
    else:
        with st.spinner("Processing files and building vector store..."):
            docs = build_documents_from_uploads(uploaded_files, chunk_size=chunk_size, chunk_overlap=chunk_overlap, max_files=max_files, max_total_chunks=max_total_chunks)
            if not docs:
                st.sidebar.error("No text extracted from uploaded files. Check files or enable Vision service account in secrets.")
            else:
                vs = build_faiss_from_documents(docs, save_local=True, faiss_path=FAISS_PATH)
                if vs:
                    st.session_state.vector_store = vs
                    st.session_state.indexed_files = [f.name for f in uploaded_files]
                    st.success(f"Vector store built. Files indexed: {len(st.session_state.indexed_files)}  | Total chunks: {st.session_state.total_chunks}")
                else:
                    st.error("Failed to create vector store.")

# --- Main: Chat UI ---
st.title("🤖 Gemini Multi-file Chatbot (Vision OCR)")
if st.session_state.indexed_files:
    st.markdown("**Indexed files:**")
    for fname in st.session_state.indexed_files:
        st.write(f"- {fname}")
    st.caption(f"Total chunks indexed (session): {st.session_state.total_chunks}")

query = st.text_input("Ask a question based on indexed documents:", placeholder="E.g. Summarize references about topic X...")
k = st.number_input("Number of retrieved chunks (k)", min_value=1, max_value=10, value=5)

if st.button("Ask") and query.strip():
    if not st.session_state.vector_store:
        st.warning("No vector store available. Upload files and build vector store first.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            hits = st.session_state.vector_store.similarity_search(query, k=k)
        if not hits:
            st.info("No relevant documents found.")
        else:
            context_block = "\n\n---\n\n".join([f"[{i+1}] ({d.metadata.get('source_file','-')}#chunk-{d.metadata.get('chunk_id','-')})\n{d.page_content}" for i,d in enumerate(hits)])
            system_instructions = ("Jawablah pertanyaan pengguna seakurat mungkin dengan mengacu pada konteks di bawah. "
                                   "Jika jawabannya tidak terdapat pada konteks, nyatakan: 'Jawaban tidak tersedia dalam konteks yang diberikan atau ditanyakan, pelajari dan berikan tambahan informasi yang diperlukan untuk memperkuat insightfull konteks atau topik yang ditanyakan dari berbagai sumber atau referensi dengan menyertakan link website.' "
                                   "Sertakan referensi [angka] ke potongan konteks bila relevan.")
            composed_prompt = f"{system_instructions}\n\n=== KONTEX ===\n{context_block}\n\n=== PERTANYAAN ===\n{query}\n\n=== JAWABAN ===\n\n=== REFERENSI/SOURCE ==="

            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
                with st.spinner("Generating answer with Gemini..."):
                    resp = llm.invoke(composed_prompt)
                out_text = getattr(resp, "content", None) or (resp.candidates[0].content if getattr(resp, "candidates", None) else str(resp))
                st.subheader("💬 Answer")
                st.write(out_text)
                with st.expander("🔎 Sources used"):
                    for i,d in enumerate(hits, start=1):
                        st.markdown(f"**[{i}]** {d.metadata.get('source_file','-')} (chunk {d.metadata.get('chunk_id','-')})")
                        st.caption(d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""))
            except Exception as e:
                st.error(f"Error calling Gemini: {e}")

