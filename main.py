import streamlit as st
from utils.pdf_handler import extract_text_from_pdfs
from utils.text_splitter import split_text
from utils.vector_store import create_vector_store
from utils.qa_chain import build_qa_chain
from utils.user_data import save_user_info

st.set_page_config(page_title="Chatbot Properti Gading Serpong", layout="wide")

def clear_chat():
    st.session_state.messages = [{"role": "assistant", "content": "Silakan upload brosur properti dan ajukan pertanyaan"}]
    st.session_state.vector_store = None

def main():
    st.sidebar.title("📁 Upload Brosur Properti")
    pdf_docs = st.sidebar.file_uploader("Upload file PDF", accept_multiple_files=True)

    if st.sidebar.button("Proses Brosur"):
        with st.spinner("Memproses dokumen..."):
            raw_text = extract_text_from_pdfs(pdf_docs)
            chunks = split_text(raw_text)
            vector_store = create_vector_store(chunks)
            st.session_state.vector_store = vector_store
            st.success("Brosur berhasil diproses!")

    st.sidebar.button("🧹 Hapus Riwayat Chat", on_click=clear_chat)

    st.title("💬 Chatbot Properti Gading Serpong")
    if "messages" not in st.session_state:
        clear_chat()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ajukan pertanyaan..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.vector_store is None:
            st.warning("Silakan upload dan proses brosur terlebih dahulu.")
            return

        docs = st.session_state.vector_store.similarity_search(prompt)
        context = "\n".join([doc.page_content for doc in docs])
        chain = build_qa_chain()
        response = chain({"input_documents": docs, "context": context, "question": prompt}, return_only_outputs=True)

        st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})
        with st.chat_message("assistant"):
            st.markdown(response["output_text"])

        if "call me" in prompt.lower():
            st.subheader("📇 Form Kontak")
            with st.form("contact_form"):
                name = st.text_input("Nama")
                phone = st.text_input("No. Telepon")
                email = st.text_input("Email")
                if st.form_submit_button("Kirim"):
                    save_user_info(name, phone, email)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Terima kasih {name}, kami akan menghubungi Anda di {phone} atau {email}."
                    })

if __name__ == "__main__":
    main()
