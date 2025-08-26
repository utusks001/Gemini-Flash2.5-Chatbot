import os
from config import GOOGLE_API_KEY
import csv
import streamlit as st
from pdf_handler import extract_text_from_pdfs
from text_splitter import split_text
from vector_store import create_vector_store
from qa_chain import build_qa_chain
from user_data import save_user_info

# 1. config.py (API Key Setup)

# 2. pdf_handler.py (PDF Reader)

# 3. text_splitter.py (Text Chunking)

# 4. vector_store.py (Embedding & FAISS)

# 5. qa_chain.py (Prompt & Gemini Chain)

# 6. user_data.py (Save User Info)

# 7. app.py (Streamlit Interface)

st.set_page_config(page_title="Gemini PDF Chatbot", layout="wide")

def clear_chat():
    st.session_state.messages = [{"role": "assistant", "content": "Upload PDFs and ask a question"}]
    st.session_state.vector_store = None

def main():
    st.sidebar.title("📁 Upload PDFs")
    pdf_docs = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True)

    if st.sidebar.button("Submit & Process"):
        with st.spinner("Processing PDFs..."):
            raw_text = extract_text_from_pdfs(pdf_docs)
            chunks = split_text(raw_text)
            vector_store = create_vector_store(chunks)
            st.session_state.vector_store = vector_store
            st.success("PDFs processed and indexed!")

    st.sidebar.button("🧹 Clear Chat", on_click=clear_chat)

    st.title("💬 Chat with Your PDFs using Gemini 2.5 Flash")
    if "messages" not in st.session_state:
        clear_chat()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.vector_store is None:
            st.warning("Please upload and process PDFs first.")
            return

        docs = st.session_state.vector_store.similarity_search(prompt)
        context = "\n".join([doc.page_content for doc in docs])
        chain = build_qa_chain()
        response = chain({"input_documents": docs, "context": context, "question": prompt}, return_only_outputs=True)

        st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})
        with st.chat_message("assistant"):
            st.markdown(response["output_text"])

        if "call me" in prompt.lower():
            st.subheader("📇 Contact Form")
            with st.form("contact_form"):
                name = st.text_input("Name")
                phone = st.text_input("Phone")
                email = st.text_input("Email")
                if st.form_submit_button("Submit"):
                    save_user_info(name, phone, email)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Thanks {name}, we’ll contact you at {phone} or {email}."
                    })

if __name__ == "__main__":
    main()
