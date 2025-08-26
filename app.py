import os
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------
# Setup & Config
# -------------------------
st.set_page_config(page_title="Gemini 2.5 Flash Chatbot (Multi-file)", page_icon="🤖", layout="wide")
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY tidak ditemukan. Tambahkan ke file .env")
    st.stop()

# Embeddings: gunakan HuggingFace agar stabil di Streamlit Cloud
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Chunking
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", " ", ""])

# -------------------------
# Utils
# -------------------------
def extract_text_from_pdf(file_bytes: BytesIO) -> str:
    """Ekstraksi teks dari PDF (best-effort)."""
    text = ""
    try:
        reader = PdfReader(file_bytes)
        for page in reader.pages:
            text += (page.extract_text() or "")
    except Exception:
        pass
    return text

def extract_text_from_txt(file_bytes: BytesIO) -> str:
    try:
        return file_bytes.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def build_documents_from_files(files) -> list[Document]:
    """Gabungkan semua file menjadi daftar Document (dengan metadata filename & index)."""
    docs: list[Document] = []
    for f in files:
        name = f.name
        file_bytes = BytesIO(f.read())
        if name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_bytes)
        elif name.lower().endswith(".txt"):
            text = extract_text_from_txt(file_bytes)
        else:
            text = ""

        if not text.strip():
            continue

        chunks = SPLITTER.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source_file": name, "chunk_id": i}
                )
            )
    return docs

def build_faiss_from_documents(docs: list[Document]) -> FAISS | None:
    if not docs:
        return None
    return FAISS.from_documents(docs, embedding=EMBEDDINGS)

def format_context(snippets: list[Document]) -> str:
    lines = []
    for i, d in enumerate(snippets, start=1):
        src = d.metadata.get("source_file", "unknown")
        cid = d.metadata.get("chunk_id", "-")
        lines.append(f"[{i}] ({src}#chunk-{cid})\n{d.page_content}")
    return "\n\n---\n\n".join(lines)

def render_sources(snippets: list[Document]):
    with st.expander("🔎 Sumber konteks yang dipakai"):
        for i, d in enumerate(snippets, start=1):
            src = d.metadata.get("source_file", "unknown")
            cid = d.metadata.get("chunk_id", "-")
            preview = d.page_content[:300].replace("\n", " ")
            st.markdown(f"**[{i}]** **{src}** (chunk {cid})")
            st.caption(preview + ("..." if len(d.page_content) > 300 else ""))

# -------------------------
# Sidebar (Upload & Build VectorStore)
# -------------------------
st.sidebar.header("📂 Upload Dokumen")
uploaded_files = st.sidebar.file_uploader(
    "Upload banyak PDF/TXT sekaligus",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

col_a, col_b = st.sidebar.columns(2)
with col_a:
    build_btn = st.button("🚀 Build Vector Store", use_container_width=True)
with col_b:
    clear_btn = st.button("🧹 Reset", use_container_width=True)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "file_list" not in st.session_state:
    st.session_state.file_list = []

if clear_btn:
    st.session_state.vector_store = None
    st.session_state.file_list = []
    st.success("Cache vector store & daftar file dibersihkan.")

if build_btn:
    if not uploaded_files:
        st.sidebar.warning("Silakan upload minimal satu file terlebih dahulu.")
    else:
        with st.spinner("Membangun vector store dari semua file..."):
            # Perlu reset pointer file karena sudah terbaca
            # Streamlit menyimpan file in-memory; baca ulang kontennya.
            # Kita panggil .getvalue() agar BytesIO dapat di-construct ulang.
            reconstructed = []
            for f in uploaded_files:
                reconstructed.append(
                    type("Uploaded", (), {"name": f.name, "read": lambda fb=f: fb.getvalue()})
                )

            docs = build_documents_from_files(reconstructed)
            vs = build_faiss_from_documents(docs)
            if vs is None:
                st.sidebar.error("Tidak ada teks valid yang berhasil diekstrak.")
            else:
                st.session_state.vector_store = vs
                st.session_state.file_list = [f.name for f in uploaded_files]
                st.sidebar.success(f"Vector store siap. File: {len(uploaded_files)} | Chunk: {len(docs)}")

# -------------------------
# Main (Chat)
# -------------------------
st.title("🤖 Gemini 2.5 Flash Chatbot — Multi-file (PDF+TXT) ")
st.caption("Embedding: sentence-transformers/all-MiniLM-L6-v2 • Vector DB: FAISS • LLM: gemini-2.5-flash")

if st.session_state.file_list:
    st.write("**Dokumen terindeks:**")
    st.write(" • " + "\n • ".join(st.session_state.file_list))

with st.container():
    prompt = st.text_input("Tanya sesuatu berdasarkan dokumen yang kamu upload:", placeholder="Contoh: ringkas semua file tentang topik X...")

col1, col2 = st.columns([1, 3])
with col1:
    ask = st.button("Tanyakan", use_container_width=True)

if ask:
    if not prompt.strip():
        st.warning("Masukkan pertanyaan terlebih dahulu.")
        st.stop()

    if st.session_state.vector_store is None:
        st.info("Belum ada vector store. Upload file dan klik **Build Vector Store** di sidebar.")
        st.stop()

    # Retrieve
    with st.spinner("🔎 Mengambil konteks dari vector store..."):
        results = st.session_state.vector_store.similarity_search(prompt, k=5)

    # Compose prompt untuk Gemini
    system_instructions = (
        "Jawablah pertanyaan pengguna seakurat mungkin dengan mengacu pada konteks di bawah. "
        "Jika jawabannya tidak terdapat pada konteks, katakan dengan jelas: "
        "\"Jawaban tidak tersedia dalam konteks yang diberikan.\" "
        "Gunakan bahasa yang ringkas dan beri referensi [angka] ke potongan konteks bila relevan."
    )
    context_block = format_context(results)
    composed_prompt = (
        f"{system_instructions}\n\n"
        f"=== KONTEXT ===\n{context_block}\n\n"
        f"=== PERTANYAAN ===\n{prompt}\n\n"
        f"=== JAWABAN ==="
    )

    # LLM: Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    try:
        with st.spinner("🤖 Gemini sedang menjawab..."):
            response = llm.invoke(composed_prompt)
        st.subheader("💬 Jawaban")
        st.write(response.content)
        render_sources(results)
    except Exception as e:
        st.error(f"❌ Error dari Gemini: {e}")

