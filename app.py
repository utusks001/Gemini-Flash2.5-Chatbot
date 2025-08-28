# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from llm_router import get_llm_response
from vector_store import build_vector_store, search_vector_store
from file_extractors import extract_text_from_file

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Multi-LLM Chatbot", layout="wide")
st.title("🤖 Multi-LLM Chatbot — Gemini, GROQ, LLaMA")

uploaded_files = st.sidebar.file_uploader("Upload files", type=["pdf", "txt", "docx", "pptx"], accept_multiple_files=True)
llm_choice = st.sidebar.selectbox("Pilih LLM", ["Gemini", "GROQ-LLaMA3", "GROQ-Mixtral"])
build_btn = st.sidebar.button("🚀 Build Vector Store")
clear_btn = st.sidebar.button("🧹 Clear Store")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if clear_btn:
    st.session_state.vector_store = None
    st.success("✅ Vector store dihapus.")

if build_btn and uploaded_files:
    with st.spinner("🔄 Memproses file..."):
        docs = []
        for f in uploaded_files:
            text = extract_text_from_file(f)
            docs.extend(text)
        st.session_state.vector_store = build_vector_store(docs)
        st.success(f"✅ {len(docs)} chunks berhasil diindeks.")

prompt = st.text_input("Tanyakan berdasarkan dokumen:")
ask_btn = st.button("Tanyakan")

if ask_btn and prompt and st.session_state.vector_store:
    with st.spinner("🔍 Mengambil konteks..."):
        context_docs = search_vector_store(st.session_state.vector_store, prompt)
        context_text = "\n\n---\n\n".join([d.page_content for d in context_docs])
        response = get_llm_response(prompt, context_text, llm_choice)
        st.subheader("💬 Jawaban")
        st.write(response)
