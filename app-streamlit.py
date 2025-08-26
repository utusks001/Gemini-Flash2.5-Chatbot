import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------------
# Load API Key
# -------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in .env file.")
    st.stop()

# -------------------------
# Embedding Loader with Fallback
# -------------------------
def load_embeddings():
    try:
        st.info("🔎 Trying Google Generative AI Embeddings...")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        st.warning(f"⚠️ Google embeddings failed: {e}\nUsing HuggingFace fallback.")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

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

    # Upload file (opsional)
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    vector_store = None
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        vector_store = get_vector_store(text)
        st.success("✅ File berhasil diproses dan dimasukkan ke Vector Store!")

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
