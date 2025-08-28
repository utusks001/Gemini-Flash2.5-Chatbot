# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.schema import Document
import tempfile

# 🔐 Load API keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 🧠 Fungsi pemilihan model
def get_llm(model_choice: str = "gemini"):
    if model_choice == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    elif model_choice == "groq":
        return ChatGroq(temperature=0.2, model_name="llama3-8b-8192")  # atau mixtral
    else:
        st.error("❌ Model tidak dikenali.")
        st.stop()

# 📄 Fungsi pemrosesan file
def process_files(uploaded_files):
    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(file.read())
            loader = TextLoader(tmp.name)
            docs.extend(loader.load())
    return docs

# 🧠 Split dan embed dokumen
def prepare_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    return vectorstore

# 🌐 UI Streamlit
st.set_page_config(page_title="Gemini & Groq Chatbot", layout="wide")
st.title("🤖 Multi-Model Property Chatbot")

with st.sidebar:
    st.subheader("⚙️ Konfigurasi")
    model_option = st.selectbox("Pilih Model LLM", ["gemini", "groq"], index=0)
    uploaded_files = st.file_uploader("📁 Upload file teks", type=["txt", "md"], accept_multiple_files=True)
    ask_btn = st.button("🚀 Proses dan Mulai Chat")

# 🚀 Proses file dan mulai chat
if ask_btn and uploaded_files:
    with st.spinner("🔄 Memproses dokumen..."):
        docs = process_files(uploaded_files)
        vectorstore = prepare_vectorstore(docs)
        retriever = vectorstore.as_retriever()
        llm = get_llm(model_option)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.success("✅ Siap untuk bertanya!")

    query = st.text_input("💬 Ajukan pertanyaan tentang dokumen:")
    if query:
        with st.spinner("🤖 Menjawab..."):
            response = qa_chain.run(query)
            st.markdown(f"**Jawaban:** {response}")


