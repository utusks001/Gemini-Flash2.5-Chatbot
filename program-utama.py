# Streamlit UI
# program-utama.py
import streamlit as st
from modules.loader import build_documents_from_uploads
from modules.embedder import build_chroma_from_documents
from modules.model import load_llm
from modules.retriever import get_retriever
from modules.chain import build_qa_chain
from modules.pdf_export import export_answer_to_pdf

st.set_page_config(page_title="Chatbot Properti Chroma", layout="wide")
st.title("📄 Chatbot Properti — Multi-file + Chroma + Gemini")

uploaded_files = st.sidebar.file_uploader(
    "Upload file (PDF, DOCX, PPTX, TXT)",
    type=["pdf", "docx", "pptx", "txt"],
    accept_multiple_files=True
)

build_btn = st.sidebar.button("🚀 Build Vector Store")

if build_btn and uploaded_files:
    with st.spinner("🔄 Memproses dan membangun vector store..."):
        docs = build_documents_from_uploads(uploaded_files)
        vector_db = build_chroma_from_documents(docs)
        retriever = get_retriever(vector_db)
        llm = load_llm()
        qa_chain = build_qa_chain(llm, retriever)
        st.session_state.qa_chain = qa_chain
        st.success(f"✅ Vector store berhasil dibuat dari {len(docs)} chunk dokumen.")

st.subheader("💬 Ajukan Pertanyaan")
query = st.text_input("Tanyakan sesuatu berdasarkan dokumen yang diupload:")

if st.button("Tanyakan") and query and "qa_chain" in st.session_state:
    with st.spinner("🤖 Menjawab..."):
        result = st.session_state.qa_chain(query)
        st.markdown("### ✅ Jawaban")
        st.write(result["result"])

        st.markdown("### 📄 Sumber Dokumen")
        for doc in result["source_documents"]:
            st.write(f"- {doc.metadata.get('source_file', 'Tidak diketahui')} (Chunk {doc.metadata.get('chunk_id')})")

        pdf_bytes = export_answer_to_pdf(result["result"])
        st.download_button("📥 Unduh Jawaban sebagai PDF", data=pdf_bytes, file_name="jawaban_chatbot.pdf", mime="application/pdf")
