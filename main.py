# main.py
import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import streamlit as st

# file parsing
from PyPDF2 import PdfReader
from docx import Document as DocxDocument  # fallback
from pptx import Presentation as PptxPresentation  # fallback
from PIL import Image

# LangChain & FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain

# Use community HuggingFace embeddings (stable)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Gemini wrapper
from langchain_google_genai import ChatGoogleGenerativeAI

# Optional OCR (pytesseract); import lazily
try:
    import pytesseract

    HAS_PYTESSERACT = True
except Exception:
    HAS_PYTESSERACT = False

# ---------------- CONFIG ----------------
INDEX_DIR = "faiss_index"
MAX_UPLOAD_FILES = 10
DEFAULT_K = 5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace model
CHAT_MODEL = "gemini-2.5-flash"

st.set_page_config(page_title="Gemini Multi-file Chatbot (FAISS + OCR)", layout="wide")

# GOOGLE_API_KEY should be stored in Streamlit secrets (.streamlit/secrets.toml) as GOOGLE_API_KEY
if "GOOGLE_API_KEY" not in st.secrets:
    st.warning("GOOGLE_API_KEY not found in Streamlit secrets. Add it under .streamlit/secrets.toml or App Settings → Secrets.")
# NOTE: ChatGoogleGenerativeAI will read key from st.secrets when created.

# ---------------- HELPERS: text extraction ----------------
def extract_text_from_pdf(bio: BytesIO) -> str:
    text = ""
    try:
        reader = PdfReader(bio)
        for p in reader.pages:
            page_text = p.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.warning(f"Failed to extract PDF text: {e}")
    return text


def extract_text_from_txt(bio: BytesIO) -> str:
    try:
        return bio.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.warning(f"Failed to read TXT: {e}")
        return ""


def extract_text_from_docx(bio: BytesIO) -> str:
    # Try Unstructured loader if available; otherwise fallback to python-docx
    try:
        # import inside to avoid ImportError at module load time
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader

        # Unstructured loader accepts path; we still prefer python-docx with file-like
        # but if Unstructured available, write temp file and use it
        tmp = Path(tempfile_name(".docx"))
        tmp.write_bytes(bio.read())
        loader = UnstructuredWordDocumentLoader(str(tmp))
        docs = loader.load()
        text = "\n".join([d.page_content for d in docs])
        tmp.unlink(missing_ok=True)
        return text
    except Exception:
        try:
            bio.seek(0)
            doc = DocxDocument(bio)
            paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
            return "\n".join(paragraphs)
        except Exception as e:
            st.warning(f"Failed to extract DOCX: {e}")
            return ""


def extract_text_from_pptx(bio: BytesIO) -> str:
    # Try Unstructured loader if available; otherwise fallback to python-pptx
    try:
        from langchain_community.document_loaders import UnstructuredPowerPointLoader

        tmp = Path(tempfile_name(".pptx"))
        tmp.write_bytes(bio.read())
        loader = UnstructuredPowerPointLoader(str(tmp))
        docs = loader.load()
        text = "\n".join([d.page_content for d in docs])
        tmp.unlink(missing_ok=True)
        return text
    except Exception:
        try:
            bio.seek(0)
            prs = PptxPresentation(bio)
            texts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text and shape.text.strip():
                        texts.append(shape.text)
            return "\n".join(texts)
        except Exception as e:
            st.warning(f"Failed to extract PPTX: {e}")
            return ""


def extract_text_from_image(bio: BytesIO) -> str:
    if not HAS_PYTESSERACT:
        st.warning("OCR requested but pytesseract not installed. Install pytesseract and system Tesseract to enable OCR.")
        return ""
    try:
        bio.seek(0)
        img = Image.open(bio).convert("RGB")
        # optionally set lang param if you'd like (e.g., lang="ind" for Indonesian trained models)
        text = pytesseract.image_to_string(img)
        return text or ""
    except Exception as e:
        st.warning(f"OCR failed: {e}")
        return ""


# small helper to create temp file names
def tempfile_name(suffix: str = "") -> str:
    import uuid, tempfile

    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


# ---------------- Build docs & FAISS ----------------
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    bio = BytesIO(raw)
    if name.endswith(".pdf"):
        return extract_text_from_pdf(bio)
    if name.endswith(".txt"):
        return extract_text_from_txt(bio)
    if name.endswith(".docx"):
        return extract_text_from_docx(bio)
    if name.endswith(".pptx"):
        return extract_text_from_pptx(bio)
    if name.endswith((".png", ".jpg", ".jpeg")):
        return extract_text_from_image(bio)
    # legacy formats
    if name.endswith(".doc") or name.endswith(".ppt"):
        st.warning(f"Legacy format {uploaded_file.name}: convert to .docx/.pptx for better extraction.")
        return ""
    st.warning(f"Unsupported file type: {uploaded_file.name}")
    return ""


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


def build_faiss_from_documents(docs: List[Document]) -> Optional[FAISS]:
    if not docs:
        return None
    vs = FAISS.from_documents(docs, EMBEDDINGS)
    return vs


# ---------------- Save / Load FAISS ----------------
def save_faiss(vs: FAISS, index_dir: str = INDEX_DIR):
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(index_dir)


def load_faiss(index_dir: str = INDEX_DIR) -> Optional[FAISS]:
    if not Path(index_dir).exists():
        return None
    try:
        # use same embeddings when reloading
        return FAISS.load_local(index_dir, EMBEDDINGS, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Could not load FAISS index from {index_dir}: {e}")
        return None


# ---------------- UI ----------------
st.title("🤖 Gemini 2.5 Flash — Multi-file Chatbot (PDF/TXT/DOCX/PPTX/Images OCR)")

st.sidebar.header("Upload & Index")
uploaded_files = st.sidebar.file_uploader(
    "Upload files (PDF, TXT, DOCX, PPTX, PNG, JPG, JPEG) — multiple allowed",
    type=["pdf", "txt", "docx", "pptx", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_faiss(INDEX_DIR)

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if st.sidebar.button("Build/Update FAISS Index"):
    if not uploaded_files:
        st.sidebar.warning("Please upload one or more files first.")
    else:
        with st.spinner("Processing files and building FAISS index..."):
            docs = build_documents_from_uploads(uploaded_files)
            if not docs:
                st.sidebar.error("No textual content extracted from uploaded files. Check files or convert to supported formats.")
            else:
                vs = build_faiss_from_documents(docs)
                if vs:
                    st.session_state.vector_store = vs
                    save_faiss(vs)
                    st.session_state.indexed_files = [f.name for f in uploaded_files]
                    st.sidebar.success(f"FAISS built & saved. Indexed files: {len(st.session_state.indexed_files)} | Chunks: {len(docs)}")
                else:
                    st.sidebar.error("Failed to build FAISS index.")

if st.sidebar.button("Load FAISS from disk (if exists)"):
    with st.spinner("Loading FAISS from disk..."):
        vs = load_faiss(INDEX_DIR)
        if vs:
            st.session_state.vector_store = vs
            st.sidebar.success("FAISS loaded into session.")
        else:
            st.sidebar.info("No FAISS index found or failed to load.")

if st.sidebar.button("Clear vector store from session"):
    st.session_state.vector_store = None
    st.session_state.indexed_files = []
    st.sidebar.info("Cleared vector store from session (disk copy untouched).")

# show indexed files
if st.session_state.get("indexed_files"):
    st.markdown("**Indexed files:**")
    st.write("\n".join(f"• {n}" for n in st.session_state.indexed_files))

st.header("Ask a question (retrieval augmented)")

query = st.text_input("Enter question based on the indexed documents:")
k = st.number_input("Number of retrieved chunks (k)", min_value=1, max_value=20, value=DEFAULT_K)

if st.button("Ask") and query and query.strip():
    if st.session_state.vector_store is None:
        st.info("No vector store loaded. Upload and build an index first (sidebar).")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            hits = st.session_state.vector_store.similarity_search(query, k=k)

        if not hits:
            st.info("No relevant chunks found.")
        else:
            context_text = "\n\n---\n\n".join(
                f"[{i+1}] ({d.metadata.get('source_file','-')}#chunk-{d.metadata.get('chunk_id','-')})\n{d.page_content}"
                for i, d in enumerate(hits)
            )

            system_instructions = (
                "Jawablah seakurat mungkin berdasarkan konteks berikut. "
                "Jika jawaban tidak ada, katakan: "
                "'Jawaban tidak tersedia dalam konteks yang diberikan atau ditanyakan, tetapi pelajari dan beri tambahan informasi yang diperlukan sesuai konteks dengan menyertakan link websitenya.' "
                "Berikan referensi [angka] ke potongan konteks bila relevan."
            )

            composed_prompt = (
                f"{system_instructions}\n\n"
                f"=== KONTEX ===\n{context_text}\n\n"
                f"=== PERTANYAAN ===\n{query}\n\n"
                f"=== JAWABAN ==="
            )

            # Prepare Gemini LLM wrapper
            chat_model = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=st.secrets.get("GOOGLE_API_KEY", None), temperature=0.2)

            try:
                with st.spinner("Generating answer with Gemini..."):
                    # try .invoke first (wrapper may vary), fallback to .generate/.generate_text
                    try:
                        resp = chat_model.invoke(composed_prompt)
                        out_text = getattr(resp, "content", None) or (resp.candidates[0].content if getattr(resp, "candidates", None) else str(resp))
                    except Exception:
                        # fallback to LangChain-style generate
                        gen_out = chat_model.generate([composed_prompt])
                        out_text = ""
                        if hasattr(gen_out, "generations"):
                            gens = gen_out.generations
                            if gens and len(gens) > 0 and len(gens[0]) > 0:
                                out_text = getattr(gens[0][0], "text", "") or str(gens[0][0])

                st.subheader("Answer")
                st.write(out_text)
                # show sources
                with st.expander("Sources / Retrieved chunks"):
                    for i, d in enumerate(hits, start=1):
                        src = d.metadata.get("source_file", "-")
                        cid = d.metadata.get("chunk_id", "-")
                        st.markdown(f"**[{i}]** {src} (chunk {cid})")
                        st.caption(d.page_content[:400] + ("..." if len(d.page_content) > 400 else ""))

            except Exception as e:
                st.error(f"Error calling Gemini: {e}")
