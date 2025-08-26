import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# -------------------------
# Load API Key
# -------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in .env file.")
    st.stop()

# -------------------------
# Embedding pakai HuggingFace
# -------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------------
# Extract text dari file
# -------------------------
def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")

    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

    return text

# -------------------------
# Helper: Create Vector Store
# -------------------------
def get_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, embedding=embeddings)

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="Gemini 2.5 Flash Chatbot", page_icon="🤖")
    st.title("🤖 Chatbot dengan Gemini 2.5 Flash")

    # Upload file
    uploaded_file = st.file_uploader("Upload a PDF atau TXT file", type=["txt", "pdf"])
    vector_store = None

    if uploaded_file is not None:
        with st.spinner("📂 Memproses file..."):
            text = extract_text_from_file(uploaded_file)
            if text.strip():
                vector_store = get_vector_store(text)
                st.success("✅ File berhasil diproses dan dimasukkan ke Vector Store!")
            else:
                st.warning("⚠️ Tidak ada teks yang bisa diekstrak dari file.")

    # Input user
    user_input = st.text_input("Masukkan pertanyaan:")
    if st.button("Kirim") and user_input:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

        if vector_store:
            docs = vector_store.similarity_search(user_input, k=3)
            context = "\n".join([d.page_content for d in docs])
            prompt = f"Gunakan konteks berikut untuk menjawab:\n\n{context}\n\nPertanyaan: {user_input}"
        else:
            prompt = user_input

        try:
            response = llm.invoke(prompt)
            st.write("### 💬 Jawaban AI:")
            st.write(response.content)
        except Exception as e:
            st.error(f"❌ Error dari Gemini: {e}")

if __name__ == "__main__":
    main()
