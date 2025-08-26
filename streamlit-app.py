import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import csv

# --- KONFIGURASI API ---
# Gunakan st.secrets untuk mengelola API key secara aman di Streamlit
# Ini adalah praktik terbaik dan menggantikan baris hardcoded 'google_api_key'
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Kunci API Google tidak ditemukan di secrets.toml. Mohon tambahkan.")
else:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"] # Set environment variable untuk LangChain

# Fungsi-fungsi lainnya tetap sama, tidak perlu diubah.
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Jawablah pertanyaan sedetail mungkin dari konteks yang diberikan, pastikan untuk memberikan semua detail. Jika jawaban tidak tersedia di konteks yang diberikan, katakan saja, "Jawaban tidak tersedia dalam konteks", jangan berikan jawaban yang salah.\n\n
    Konteks:\n {context}?\n
    Pertanyaan: \n{question}\n

    Jawaban:
    """
    # Perbaikan: Hapus parameter 'client'. LangChain secara internal sudah mengaturnya.
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
                                   temperature=0.3)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Unggah beberapa PDF dan ajukan pertanyaan kepada saya"}]

def user_input(user_question):
    # Perbaikan: Tidak perlu membuat embeddings baru setiap kali.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True,
    )
    return response['output_text']

def save_user_info(name, phone, email):
    file_exists = os.path.isfile('user_info.csv')
    with open('user_info.csv', mode='a', newline='') as file:
        fieldnames = ['Name', 'Phone', 'Email']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'Name': name, 'Phone': phone, 'Email': email})

def main():
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🖐",
        layout="wide"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Unggah file PDF Anda dan klik tombol Proses", accept_multiple_files=True)
        if st.button("Proses"):
            with st.spinner("Sedang memproses..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Selesai")
                else:
                    st.error("Mohon unggah file PDF terlebih dahulu.")

    # Main content area for displaying chat messages
    st.title("Chat dengan file PDF menggunakan Gemini 🙋‍♂️")
    st.write("Selamat datang!")
    st.sidebar.button('Bersihkan Riwayat Chat', on_click=clear_chat_history)

    # Chat input
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Unggah beberapa PDF dan ajukan pertanyaan kepada saya"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check for specific user request to call them
        if "hubungi saya" in prompt.lower() or "call me" in prompt.lower():
            st.session_state.collecting_info = True

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Sedang berpikir..."):
                    if 'faiss_index' in os.listdir('.'):
                        response = user_input(prompt)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.markdown(response)
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "Mohon unggah dan proses file PDF terlebih dahulu."})
                        st.markdown("Mohon unggah dan proses file PDF terlebih dahulu.")

    # Collect user information
    if "collecting_info" in st.session_state and st.session_state.collecting_info:
        st.subheader("Mohon berikan detail kontak Anda:")
        with st.form(key="contact_form"):
            name = st.text_input("Nama")
            phone = st.text_input("Nomor Telepon")
            email = st.text_input("Email")
            submit_button = st.form_submit_button(label="Kirim")

            if submit_button:
                save_user_info(name, phone, email)
                st.session_state.messages.append({"role": "assistant", "content": f"Terima kasih, {name}. Kami akan menghubungi Anda di {phone} atau {email}."})
                st.session_state.collecting_info = False

if __name__ == "__main__":
    main()
