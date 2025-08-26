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
# Extract text dari file (PDF / TXT)
# -------------------------
def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

    return text

# -------------------------
# Create Vector Store dari banyak file
# -------------------------
def get_vector_store_from_files(uploaded_files):
    all_text = ""
    for file in uploaded_files:
        extracted_text = extract_text_from_file(file)
        all_text += extracted_text + "\n"

    if not all_text.strip():
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(all_text)
    return FAISS.from_texts(chunks, embedding=embeddings)

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="Gemini 2.5 Flash Multi-file Chatbot", page_icon="🤖")
    st.title("🤖 Chatbot dengan Gemini 2.5 Flash (Multi-file Upload)")

    # Upload multiple files
    uploaded_files = st.file_uploader("Upload PDF atau TXT (boleh lebih dari 1 file)", type=["txt", "pdf"], accept_multiple_files=True)
    vector_store = None

    if uploaded_files:
        with st.spinner("📂 Memproses semua file..."):
            vector_store = get_vector_store_from_files(uploaded_files)
            if vector_store:
                st.success(f"✅ {len(uploaded_files)} file berhasil diproses dan dimasukkan ke Vector Store!")
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
