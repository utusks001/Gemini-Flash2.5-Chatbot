# Streamlit UI
# program-utama.py
"""
Gemini / Groq Image-enabled Multi-File Chatbot
================================================
Streamlit app yang bisa menerima file PDF, DOCX, PPTX, TXT **dan** gambar,
ekstraksi teks (OCR utk gambar), buat FAISS vector store, lalu jawab pertanyaan
pakai Gemini atau Groq 
"""
import os
import streamlit as st
import pytesseract
from PIL import Image
from io import BytesIO

from PyPDF2 import PdfReader
import docx
import pptx

# LangChain imports
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Google GenAI LangChain wrapper
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
except Exception:
    GoogleGenerativeAIEmbeddings = None
    ChatGoogleGenerativeAI = None

# ---------------------------
# Config / Secrets
# ---------------------------
# Streamlit Cloud: set secrets in Settings → Secrets
GOOGLE_API_KEY = None
try:
    # prefer st.secrets (Streamlit Cloud). Fallback to env for local dev.
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
except Exception:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# If you want to support Groq too, you can add similar check for GROQ_API_KEY here.

# ---------------------------
# File readers / OCR helpers
# ---------------------------
def read_pdf(file_like):
    try:
        pdf = PdfReader(file_like)
        text_parts = []
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        st.warning(f"⚠️ Gagal membaca PDF: {e}")
        return ""

def read_docx(file_like):
    try:
        doc = docx.Document(file_like)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        st.warning(f"⚠️ Gagal membaca DOCX: {e}")
        return ""

def read_pptx(file_like):
    try:
        prs = pptx.Presentation(file_like)
        texts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    texts.append(shape.text)
        return "\n".join(texts)
    except Exception as e:
        st.warning(f"⚠️ Gagal membaca PPTX: {e}")
        return ""

def read_txt(file_like):
    try:
        raw = file_like.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return str(raw)
    except Exception as e:
        st.warning(f"⚠️ Gagal membaca TXT: {e}")
        return ""

def read_image(file_like):
    try:
        # file_like is a Streamlit UploadedFile (supports read/seek)
        image = Image.open(file_like).convert("RGB")
        # pytesseract accepts PIL Image
        text = pytesseract.image_to_string(image, lang="eng+ind")
        return text
    except Exception as e:
        st.warning(f"⚠️ OCR image failed: {e}")
        return ""

# ---------------------------
# Build documents from uploads
# ---------------------------
def build_documents_from_uploads(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        name = uploaded_file.name.lower()
        text = ""

        # ensure file pointer at start
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        if name.endswith(".pdf"):
            text = read_pdf(uploaded_file)
        elif name.endswith(".docx"):
            text = read_docx(uploaded_file)
        elif name.endswith(".pptx"):
            text = read_pptx(uploaded_file)
        elif name.endswith(".txt"):
            text = read_txt(uploaded_file)
        elif name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".jfif")):
            # for images, make a copy of bytes for PIL
            # Streamlit's UploadedFile is file-like; PIL can open it directly
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
            text = read_image(uploaded_file)
        else:
            st.warning(f"⚠️ Unsupported file type: {uploaded_file.name}")
            continue

        if text and text.strip():
            docs.append(Document(page_content=text, metadata={"source": uploaded_file.name}))
    return docs

# ---------------------------
# Build / use vectorstore
# ---------------------------
def build_vector_store(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splitted = splitter.split_documents(documents)
    if not GOOGLE_API_KEY:
        st.error("❌ GOOGLE_API_KEY belum di-set. Masukkan di Streamlit Secrets (GOOGLE_API_KEY).")
        return None
    if GoogleGenerativeAIEmbeddings is None:
        st.error("❌ Paket langchain_google_genai tidak tersedia. Pastikan requirements diinstall.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    # FAISS.from_documents expects keyword 'embedding' in some versions
    try:
        vs = FAISS.from_documents(splitted, embedding=embeddings)
    except TypeError:
        # fallback: try positional
        vs = FAISS.from_documents(splitted, embeddings)
    return vs

# ---------------------------
# UI / Main
# ---------------------------
def main():
    st.set_page_config(page_title="Multi-file RAG Chatbot (Tesseract + Gemini)", layout="wide")
    st.title("📚 Multi-file RAG Chatbot — OCR (Tesseract) + Gemini")

    st.markdown(
        "Upload file PDF / DOCX / PPTX / TXT / Images (PNG, JPG, JPEG, BMP, GIF, JFIF). "
        "Images will be OCR'ed with Tesseract (eng+ind)."
    )

    uploaded_files = st.file_uploader(
        "Upload files (multiple)",
        type=["pdf", "docx", "pptx", "txt", "png", "jpg", "jpeg", "bmp", "gif", "jfif"],
        accept_multiple_files=True
    )

    # Build button & reset
    if "vs" not in st.session_state:
        st.session_state.vs = None
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Build Vector Store") and uploaded_files:
            with st.spinner("Processing files and building index..."):
                docs = build_documents_from_uploads(uploaded_files)
                if not docs:
                    st.error("Tidak ada teks yang berhasil diekstrak dari file.")
                else:
                    vs = build_vector_store(docs)
                    if vs:
                        st.session_state.vs = vs
                        st.session_state.indexed_files = [f.name for f in uploaded_files]
                        st.success(f"Vector store terbangun — dokumen terindeks: {len(st.session_state.indexed_files)} | chunks: {len(docs)}")

        if st.button("🧹 Reset index"):
            st.session_state.vs = None
            st.session_state.indexed_files = []
            st.success("Index di-reset.")

    with col2:
        if st.session_state.indexed_files:
            st.markdown("**Dokumen terindeks:**")
            st.write("\n • ".join(st.session_state.indexed_files))

    # Querying
    st.subheader("🗣️ Tanya sesuatu (berdasarkan dokumen terindeks)")
    prompt = st.text_input("Masukkan pertanyaan:")
    if st.button("Tanyakan") and prompt:
        if st.session_state.vs is None:
            st.info("Belum ada vector store. Upload file dan klik 'Build Vector Store' terlebih dahulu.")
        else:
            with st.spinner("Mencari konteks..."):
                try:
                    results = st.session_state.vs.similarity_search(prompt, k=5)
                except Exception:
                    # fallback to retriever method
                    retriever = st.session_state.vs.as_retriever(search_kwargs={"k": 5})
                    results = retriever.get_relevant_documents(prompt)

                if not results:
                    st.warning("Tidak ditemukan konteks relevan.")
                else:
                    context_text = "\n\n---\n\n".join([f"[{i+1}] ({d.metadata.get('source','unknown')})\n{d.page_content}" for i, d in enumerate(results)])
                    system_instructions = (
                        "Jawablah seakurat dan sedetil mungkin berdasarkan konteks berikut. "
                        "Jika jawaban tidak ada, katakan: \"Jawaban tidak tersedia dalam konteks yang diberikan.\""
                    )
                    composed_prompt = f"{system_instructions}\n\n=== KONTEX ===\n{context_text}\n\n=== PERTANYAAN ===\n{prompt}\n\n=== JAWABAN ==="

                    # choose LLM
                    if ChatGoogleGenerativeAI is None:
                        st.error("❌ Paket langchain_google_genai tidak terpasang — LLM tidak tersedia.")
                        return

                    try:
                        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.0)
                        response = llm.invoke(composed_prompt)
                        out_text = getattr(response, "content", None) or str(response)
                    except Exception as e:
                        st.error(f"❌ Error saat memanggil LLM: {e}")
                        out_text = None

                    if out_text:
                        st.subheader("💬 Jawaban")
                        st.write(out_text)
                        with st.expander("🔎 Sumber konteks (potongan)"):
                            for i, d in enumerate(results, start=1):
                                src = d.metadata.get("source", "unknown")
                                preview = d.page_content[:300].replace("\n", " ")
                                st.markdown(f"**[{i}]** **{src}**")
                                st.caption(preview + ("..." if len(d.page_content) > 300 else ""))
                    else:
                        st.error("LLM tidak menghasilkan output.")

if __name__ == "__main__":
    main()

