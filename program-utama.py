# Streamlit UI
# program-utama.py

"""
app.py
Multi-file RAG Chatbot (OCR.space + Gemini / optional Groq)
- No type hints anywhere (safe for Streamlit Cloud)
- OCR for images via OCR.space API
- Supports PDF, DOCX, PPTX, TXT, Images
- LLM provider selectable: Gemini (Google) or Groq (if GROQ_API_KEY provided)
"""

import os
import streamlit as st
import requests
from io import BytesIO

# file parsing
from PyPDF2 import PdfReader
import docx
import pptx
from PIL import Image

# langchain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# try to import wrappers (may be missing depending on your env)
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
except Exception:
    GoogleGenerativeAIEmbeddings = None
    ChatGoogleGenerativeAI = None

try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

# try to import huggingface embeddings as fallback
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None

# -----------------------
# Config / secrets
# -----------------------
# In Streamlit Cloud, put keys in Settings -> Secrets (secrets.toml)
OCRSPACE_API_KEY = st.secrets.get("OCRSPACE_API_KEY") or os.environ.get("OCRSPACE_API_KEY")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

# Set env for libraries that might read it
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# -----------------------
# Helpers: read different file types
# -----------------------
def read_pdf(file_like):
    try:
        reader = PdfReader(file_like)
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception as e:
        st.warning(f"Failed reading PDF: {e}")
        return ""

def read_docx(file_like):
    try:
        doc = docx.Document(file_like)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        st.warning(f"Failed reading DOCX: {e}")
        return ""

def read_pptx(file_like):
    try:
        prs = pptx.Presentation(file_like)
        parts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    parts.append(shape.text)
        return "\n".join(parts)
    except Exception as e:
        st.warning(f"Failed reading PPTX: {e}")
        return ""

def read_txt(file_like):
    try:
        raw = file_like.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return str(raw)
    except Exception as e:
        st.warning(f"Failed reading TXT: {e}")
        return ""

def read_image_ocr_via_ocrspace(file_like):
    """
    Uses OCR.space API to parse image bytes.
    Requires OCRSPACE_API_KEY in secrets or env.
    """
    if not OCRSPACE_API_KEY:
        st.warning("OCR.space API key not set (OCRSPACE_API_KEY). Image OCR skipped.")
        return ""
    try:
        # Prepare file bytes
        file_like.seek(0)
        file_bytes = file_like.read()
        # Build request
        url = "https://api.ocr.space/parse/image"
        payload = {
            "apikey": OCRSPACE_API_KEY,
            "language": "eng",  # change to "eng+ind" if OCR.space supports combined code or use "ind" separately
            "isOverlayRequired": False,
            "OCREngine": 2  # optional: OCR engine selection
        }
        files = {"file": ("image", file_bytes)}
        resp = requests.post(url, data=payload, files=files, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        if result.get("IsErroredOnProcessing"):
            err = result.get("ErrorMessage") or result.get("ErrorDetails")
            st.warning(f"OCR.space error: {err}")
            return ""
        parsed = result.get("ParsedResults", [])
        text_parts = [p.get("ParsedText", "") for p in parsed]
        return "\n".join(text_parts).strip()
    except Exception as e:
        st.warning(f"OCR.space request failed: {e}")
        return ""

# -----------------------
# Build documents (no type hints)
# -----------------------
def build_documents_from_uploads(uploaded_files):
    docs = []
    for f in uploaded_files:
        # make sure pointer at start
        try:
            f.seek(0)
        except Exception:
            pass

        name = f.name.lower()
        text = ""

        if name.endswith(".pdf"):
            text = read_pdf(f)
        elif name.endswith(".docx"):
            text = read_docx(f)
        elif name.endswith(".pptx"):
            text = read_pptx(f)
        elif name.endswith(".txt"):
            text = read_txt(f)
        elif name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".jfif")):
            text = read_image_ocr_via_ocrspace(f)
        else:
            st.warning(f"Unsupported file type: {f.name}")
            continue

        if text and text.strip():
            docs.append(Document(page_content=text, metadata={"source": f.name}))
    return docs

# -----------------------
# Build / load vectorstore
# -----------------------
def build_vectorstore(documents):
    if not documents:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    split_docs = splitter.split_documents(documents)

    # prefer Google embeddings if API key available and wrapper installed
    if GOOGLE_API_KEY and GoogleGenerativeAIEmbeddings is not None:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
            vs = FAISS.from_documents(split_docs, embedding=embeddings)
            return vs
        except Exception as e:
            st.warning(f"Google embeddings failed: {e}")

    # fallback: HuggingFace embeddings if available
    if HuggingFaceEmbeddings is not None:
        try:
            hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vs = FAISS.from_documents(split_docs, embedding=hf)
            st.info("Using HuggingFace embeddings (fallback).")
            return vs
        except Exception as e:
            st.warning(f"HuggingFace embeddings failed: {e}")

    st.error("No usable embeddings available. Set GOOGLE_API_KEY or install HuggingFace embeddings.")
    return None

# -----------------------
# LLM helper (create LLM instance based on selection)
# -----------------------
def get_llm_instance(choice, model_name):
    # choice: "Gemini" or "Groq"
    if choice == "Groq":
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY not set in secrets")
            return None
        if ChatGroq is None:
            st.error("langchain_groq not installed or importable")
            return None
        try:
            llm = ChatGroq(model=model_name, groq_api_key=GROQ_API_KEY, temperature=0.0)
            return llm
        except Exception as e:
            st.error(f"Failed to init Groq LLM: {e}")
            return None
    else:
        # Gemini / Google
        if not GOOGLE_API_KEY:
            st.error("GOOGLE_API_KEY not set in secrets")
            return None
        if ChatGoogleGenerativeAI is None:
            st.error("langchain_google_genai not installed or importable")
            return None
        try:
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY, temperature=0.0)
            return llm
        except Exception as e:
            st.error(f"Failed to init Google LLM: {e}")
            return None

# -----------------------
# Streamlit UI
# -----------------------
def main():
    st.set_page_config(page_title="Multi-file RAG (OCR.space) — Gemini / Groq", layout="wide")
    st.title("📚 Multi-file RAG Chatbot — OCR.space + Gemini/Groq")

    st.markdown("Upload PDF / DOCX / PPTX / TXT / Images. Images are OCR'ed with OCR.space (needs OCRSPACE_API_KEY).")

    # Sidebar: LLM selection
    with st.sidebar:
        st.header("Settings")
        provider_options = ["Gemini"]
        if GROQ_API_KEY:
            provider_options.append("Groq")
        provider = st.radio("LLM provider", provider_options, index=0)

        if provider == "Groq":
            model_name = st.selectbox("Groq model", options=["mixtral-8x7b-32768", "llama-3.1-70b-versatile"])
        else:
            model_name = st.selectbox("Gemini model", options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash"])

        st.markdown("**Note:** Make sure required API keys are set in Streamlit Secrets.")

    uploaded_files = st.file_uploader(
        "Upload files (multiple):",
        type=["pdf", "docx", "pptx", "txt", "png", "jpg", "jpeg", "bmp", "gif", "jfif"],
        accept_multiple_files=True
    )

    # session state for vector store and indexed filenames
    if "vs" not in st.session_state:
        st.session_state.vs = None
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Build Vector Store") and uploaded_files:
            with st.spinner("Processing files and building vector store..."):
                documents = build_documents_from_uploads(uploaded_files)
                if not documents:
                    st.error("No text extracted from uploads.")
                else:
                    vs = build_vectorstore(documents)
                    if vs:
                        st.session_state.vs = vs
                        st.session_state.indexed_files = [f.name for f in uploaded_files]
                        st.success(f"Vector store built. Indexed files: {len(st.session_state.indexed_files)}")
        if st.button("Reset Index"):
            st.session_state.vs = None
            st.session_state.indexed_files = []
            st.success("Index reset.")

    with col2:
        if st.session_state.indexed_files:
            st.markdown("**Indexed files:**")
            for name in st.session_state.indexed_files:
                st.write("- " + name)

    st.markdown("---")
    st.subheader("Ask / Query")

    query = st.text_input("Enter your question (based on indexed documents):")
    if st.button("Ask") and query:
        if st.session_state.vs is None:
            st.info("No vector store yet. Upload files and click 'Build Vector Store'.")
        else:
            with st.spinner("Retrieving context..."):
                try:
                    docs = st.session_state.vs.similarity_search(query, k=5)
                except Exception:
                    retr = st.session_state.vs.as_retriever(search_kwargs={"k": 5})
                    docs = retr.get_relevant_documents(query)

            if not docs:
                st.warning("No relevant context found.")
            else:
                # compose context
                context_blocks = []
                for i, d in enumerate(docs, start=1):
                    src = d.metadata.get("source", "unknown")
                    preview = d.page_content[:400].replace("\n", " ")
                    context_blocks.append(f"[{i}] ({src})\n{preview}")
                context_text = "\n\n---\n\n".join(context_blocks)

                prompt_system = (
                    "Jawablah seakurat dan sedetil mungkin berdasarkan konteks berikut. "
                    "Jika jawaban tidak ada, katakan: \"Jawaban tidak tersedia dalam konteks yang diberikan.\""
                )
                full_prompt = f"{prompt_system}\n\n=== KONTEX ===\n{context_text}\n\n=== PERTANYAAN ===\n{query}\n\n=== JAWABAN ==="

                llm = get_llm_instance(provider, model_name)
                if llm is None:
                    st.error("LLM instance could not be created. Check API keys / installed packages.")
                else:
                    with st.spinner("Calling LLM..."):
                        try:
                            # prefer invoke if available, else try generate/predict
                            if hasattr(llm, "invoke"):
                                resp = llm.invoke(full_prompt)
                                out = getattr(resp, "content", None) or str(resp)
                            elif hasattr(llm, "predict"):
                                out = llm.predict(full_prompt)
                            elif hasattr(llm, "generate"):
                                gen = llm.generate(full_prompt)
                                out = str(gen)
                            else:
                                out = "LLM returned no usable method."
                        except Exception as e:
                            out = f"LLM call failed: {e}"

                    st.subheader("Answer")
                    st.write(out)

                    with st.expander("Retrieved context (full chunks)"):
                        for i, d in enumerate(docs, start=1):
                            st.markdown(f"**[{i}] {d.metadata.get('source','unknown')}**")
                            st.write(d.page_content[:2000] + ("..." if len(d.page_content) > 2000 else ""))

    # End main

if __name__ == "__main__":
    main()
