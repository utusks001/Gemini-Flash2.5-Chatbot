# streamlit-app.py
# Gemini Flash 2.5 Chatbot - Upload Multi-files, Build Vector Store and Asking (with Groq Support)
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set API keys
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# UI
st.set_page_config(page_title="📚 Multi-Model Chatbot", layout="wide")
st.title("📚 Chatbot Multi-Model: Gemini 2.5 Flash + Groq")

# Sidebar: Model selection
model_choice = st.sidebar.selectbox("Pilih Model:", ["gemini-2.5-flash", "llama3-8b-8192", "mixtral-8x7b-32768"])

# Upload files
uploaded_files = st.file_uploader("📁 Upload file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Build vector store
if st.button("⚙️ Build Vector Store") and uploaded_files:
    docs = []
    for file in uploaded_files:
        ext = file.name.split(".")[-1].lower()
        if ext == "pdf":
            loader = PyPDFLoader(file)
        elif ext == "docx":
            loader = Docx2txtLoader(file)
        elif ext == "txt":
            loader = TextLoader(file)
        else:
            st.warning(f"Format tidak didukung: {file.name}")
            continue
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    st.session_state.vector_store = vector_store
    st.success("✅ Vector store berhasil dibuat!")

# Prompt input
prompt = st.text_area("💬 Masukkan pertanyaan:")
ask_btn = st.button("🚀 Jawab")

# LLM selector
def get_llm(model_name):
    if model_name.startswith("gemini"):
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    elif model_name.startswith(("llama3", "mixtral")):
        return ChatGroq(model_name=model_name, temperature=0.2)
    else:
        raise ValueError("Model tidak dikenali.")

# Format context
def format_context(docs):
    return "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])

# Ask button logic
if ask_btn:
    if not prompt.strip():
        st.warning("Masukkan pertanyaan terlebih dahulu.")
    elif "vector_store" not in st.session_state:
        st.warning("Vector store belum dibuat.")
    else:
        results = st.session_state.vector_store.similarity_search(prompt, k=5)
        context_text = format_context(results)

        system_prompt = (
            "Jawablah seakurat dan sedetil mungkin berdasarkan konteks berikut. "
            "Jika jawaban tidak tersedia, katakan: "
            "\"Jawaban tidak tersedia dalam konteks yang diberikan, pelajari dan berikan tambahan informasi yang diperlukan sesuai konteks atau topik yang ditanyakan dari berbagai sumber atau referensi di internet dengan menyertakan link website.\" "
            "Sertakan referensi [angka] ke potongan konteks bila relevan."
        )

        full_prompt = f"{system_prompt}\n\n=== KONTEX ===\n{context_text}\n\n=== PERTANYAAN ===\n{prompt}\n\n=== JAWABAN ==="

        try:
            llm = get_llm(model_choice)
            response = llm.invoke(full_prompt)
            out_text = getattr(response, "content", None) or (
                response.candidates[0].content if getattr(response, "candidates", None) else str(response)
            )
            st.subheader("💬 Jawaban")
            st.write(out_text)
            st.markdown("📌 Referensi:")
            for i, doc in enumerate(results):
                st.markdown(f"[{i+1}] {doc.metadata.get('source', 'Tidak diketahui')}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
