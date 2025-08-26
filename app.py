# app.py
import os
import json
from io import BytesIO
from typing import List
import base64
import time

import streamlit as st
from dotenv import load_dotenv
import requests
from PIL import Image
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation as PptxPresentation
import fitz  # pymupdf

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# Optional pytesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# Load env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # for Gemini (if used)
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")  # for Vision OCR

st.set_page_config(page_title="Gemini Multi-file Chatbot + Vision OCR + FAISS", page_icon="🤖", layout="wide")

# Basic checks
if not GOOGLE_API_KEY:
    st.warning("GOOGLE_API_KEY not set. Gemini calls will fail unless provided. You can still build index + use embeddings locally.")
if not GOOGLE_VISION_API_KEY:
    st.info("GOOGLE_VISION_API_KEY not set. Image OCR will try local pytesseract if available; otherwise images won't be OCR'd. To enable OCR in Streamlit Cloud, set GOOGLE_VISION_API_KEY in .env.")

# Embeddings & splitter default
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Session keys
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "faiss_path" not in st.session_state:
    st.session_state.faiss_path = "faiss_index"  # default local dir

# ---------- Helpers: OCR via Google Vision REST ----------
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

def ocr_image_with_google_vision(image_bytes: bytes, api_key: str) -> str:
    """
    Call Google Vision API (DOCUMENT_TEXT_DETECTION) via REST.
    Returns extracted text (string).
    """
    img_base64 = base64.b64encode(image_bytes).decode("utf-8")
    headers = {"Content-Type": "application/json"}
    body = {
        "requests": [
            {
                "image": {"content": img_base64},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION", "maxResults": 1}],
                "imageContext": {}
            }
        ]
    }
    params = {"key": api_key}
    try:
        resp = requests.post(VISION_ENDPOINT, params=params, headers=headers, data=json.dumps(body), timeout=60)
        resp.raise_for_status()
        res_json = resp.json()
        text = ""
        # Navigate JSON
        try:
            text = res_json["responses"][0].get("fullTextAnnotation", {}).get("text", "") or ""
        except Exception:
            text = ""
        return text
    except Exception as e:
        st.warning(f"Google Vision OCR failed: {e}")
        return ""

def ocr_image_with_pytesseract(image_bytes: bytes) -> str:
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        return pytesseract.image_to_string(pil)
    except Exception:
        return ""

def ocr_image_best_effort(image_bytes: bytes) -> str:
    # prefer Vision API if key present
    if GOOGLE_VISION_API_KEY:
        txt = ocr_image_with_google_vision(image_bytes, GOOGLE_VISION_API_KEY)
        if txt and txt.strip():
            return txt
    # fallback to local tesseract if available
    if TESSERACT_AVAILABLE:
        return ocr_image_with_pytesseract(image_bytes)
    return ""

# ---------- File text extractors ----------
def extract_text_from_pdf_bytes(raw_bytes: bytes, use_ocr: bool = True) -> str:
    text = ""
    # 1. Try PyPDF2 text extraction
    try:
        reader = PdfReader(BytesIO(raw_bytes))
        for page in reader.pages:
            ptext = page.extract_text()
            if ptext:
                text += ptext + "\n"
    except Exception:
        pass

    # 2. If empty and use_ocr True -> render pages via PyMuPDF and OCR pages
    if (not text.strip()) and use_ocr:
        try:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes()
                ocr_text = ocr_image_best_effort(img_bytes)
                if ocr_text:
                    text += ocr_text + "\n"
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
        # images in slides -> OCR if needed
        if use_ocr:
            for slide in prs.slides:
                for shape in slide.shapes:
                    try:
                        if shape.shape_type == 13:  # picture
                            img = shape.image
                            img_bytes = img.blob
                            ocr_text = ocr_image_best_effort(img_bytes)
                            if ocr_text:
                                text += ocr_text + "\n"
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
    # Direct OCR of image file
    return ocr_image_best_effort(raw_bytes)

# ---------- Build docs & FAISS ----------
def build_documents_from_uploads(uploaded_files, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP, max_files=20, max_total_chunks=2000) -> List[Document]:
    """
    Args:
      uploaded_files: list of UploadedFile objects
      returns list of langchain Documents (with metadata)
    """
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
            st.warning(f"File `{fname}` is legacy (.doc/.ppt). Convert to .docx/.pptx for better extraction.")
            text = ""
        else:
            st.warning(f"Unsupported file type: {fname}")
            text = ""

        if not text or not text.strip():
            file_count += 1
            continue

        chunks = splitter.split_text(text)
        # enforce max total chunks to avoid OOM
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

def build_faiss_from_documents(docs: List[Document], save_local: bool=True, faiss_path: str="faiss_index") -> FAISS | None:
    if not docs:
        return None
    vs = FAISS.from_documents(docs, embedding=EMBEDDINGS)
    if save_local:
        try:
            vs.save_local(faiss_path)
        except Exception as e:
            st.warning(f"Could not save FAISS to disk: {e}")
    return vs

def load_faiss_if_exists(faiss_path="faiss_index"):
    try:
        if os.path.exists(faiss_path):
            vs = FAISS.load_local(faiss_path, EMBEDDINGS, allow_dangerous_deserialization=True)
            return vs
    except Exception as e:
        st.warning(f"Failed to load local FAISS index: {e}")
    return None

# ---------- UI: Sidebar settings ----------
st.sidebar.header("Settings & Upload")
max_files = st.sidebar.number_input("Max files to index (per build)", min_value=1, max_value=50, value=20)
max_total_chunks = st.sidebar.number_input("Max total chunks (per build)", min_value=100, max_value=20000, value=2000, step=100)
chunk_size = st.sidebar.number_input("Chunk size (characters)", min_value=200, max_value=5000, value=DEFAULT_CHUNK_SIZE, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap (characters)", min_value=0, max_value=1000, value=DEFAULT_CHUNK_OVERLAP, step=10)

uploaded_files = st.sidebar.file_uploader(
    "Upload multiple files (pdf, txt, docx, pptx, images...)", 
    type=["pdf", "txt", "docx", "pptx", "png", "jpg", "jpeg", "jfif", "gif", "bmp", "webp", "tiff"], 
    accept_multiple_files=True
)

col1, col2, col3 = st.sidebar.columns(3)
build_btn = col1.button("Build Vector Store")
load_btn = col2.button("Load saved FAISS")
clear_btn = col3.button("Clear vector store")

if clear_btn:
    st.session_state.vector_store = None
    st.session_state.indexed_files = []
    st.session_state.total_chunks = 0
    # optionally remove disk data
    try:
        if os.path.exists(st.session_state.faiss_path):
            import shutil
            shutil.rmtree(st.session_state.faiss_path)
    except Exception:
        pass
    st.success("Cleared vector store (session + disk)")

if load_btn:
    vs = load_faiss_if_exists(st.session_state.faiss_path)
    if vs:
        st.session_state.vector_store = vs
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
                st.sidebar.error("No text extracted from uploaded files. Check files or enable Vision API for OCR.")
            else:
                vs = build_faiss_from_documents(docs, save_local=True, faiss_path=st.session_state.faiss_path)
                if vs:
                    st.session_state.vector_store = vs
                    st.session_state.indexed_files = [f.name for f in uploaded_files]
                    st.success(f"Vector store built. Files indexed: {len(st.session_state.indexed_files)}  | Total chunks: {st.session_state.total_chunks}")
                else:
                    st.error("Failed to create vector store.")

# show indexed files summary
if st.session_state.indexed_files:
    st.markdown("**Indexed files:**")
    for fname in st.session_state.indexed_files:
        st.write(f"- {fname}")
    st.caption(f"Total chunks indexed (session): {st.session_state.total_chunks}")

# ---------- Main Chat UI ----------
st.title("🤖 Gemini Multi-file Chatbot + Google Vision OCR + FAISS")
if GOOGLE_VISION_API_KEY:
    st.caption("OCR: using Google Vision API (DOCUMENT_TEXT_DETECTION).")
elif TESSERACT_AVAILABLE:
    st.caption("OCR: using local Tesseract (pytesseract).")
else:
    st.caption("OCR not enabled. Provide GOOGLE_VISION_API_KEY in .env to enable OCR for images & image-only PDFs.")

query = st.text_input("Ask a question based on indexed documents:", placeholder="Example: Summarize all findings about topic X...")
k = st.number_input("Number of retrieved chunks (k)", min_value=1, max_value=20, value=5)

if st.button("Ask") and query.strip():
    if st.session_state.vector_store is None:
        st.warning("No vector store available. Upload files and build the vector store first.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            hits = st.session_state.vector_store.similarity_search(query, k=k)
        if not hits:
            st.info("No relevant documents found.")
        else:
            # Format context
            ctx_parts = []
            for i, d in enumerate(hits, start=1):
                src = d.metadata.get("source_file", "unknown")
                cid = d.metadata.get("chunk_id", "-")
                ctx_parts.append(f"[{i}] ({src}#chunk-{cid})\n{d.page_content}")
            context_block = "\n\n---\n\n".join(ctx_parts)

            system_instructions = (
                "Jawablah pertanyaan pengguna seakurat mungkin dengan mengacu pada konteks di bawah. "
                "Jika jawabannya tidak terdapat pada konteks, nyatakan: 'Jawaban tidak tersedia dalam konteks yang diberikan.' "
                "Sertakan referensi [angka] ke potongan konteks bila relevan."
            )
            composed_prompt = f"{system_instructions}\n\n=== KONTEX ===\n{context_block}\n\n=== PERTANYAAN ===\n{query}\n\n=== JAWABAN ==="

            # Call Gemini (LLM)
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
                with st.spinner("Generating answer with Gemini..."):
                    resp = llm.invoke(composed_prompt)
                # get text
                out_text = getattr(resp, "content", None) or (resp.candidates[0].content if getattr(resp, "candidates", None) else str(resp))
                st.subheader("💬 Answer")
                st.write(out_text)

                # Show sources
                with st.expander("🔎 Sources used"):
                    for i, d in enumerate(hits, start=1):
                        st.markdown(f"**[{i}]** {d.metadata.get('source_file','-')} (chunk {d.metadata.get('chunk_id','-')})")
                        st.caption(d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""))
            except Exception as e:
                st.error(f"Error calling Gemini: {e}")

# ---------- Footer: tips ----------
st.markdown("---")
st.markdown("**Tips:**")
st.markdown("- Set `GOOGLE_VISION_API_KEY` in `.env` to enable OCR in Streamlit Cloud (Vision API DOCUMENT_TEXT_DETECTION).")
st.markdown("- Use moderate chunk size and max_total_chunks to avoid memory issues. Save FAISS to disk and use `Load saved FAISS` to reuse index across restarts.")
st.markdown("- For production, consider storing FAISS on S3/GCS and loading it on startup to persist indexes.")

