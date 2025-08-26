import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import csv

# Load environment
#load_dotenv()
#os.environ["GOOGLE_API_KEY"] = "AIzaSyDja-EvridYPmwLkxqYx9DD6Tkv4WAfbJk"

# Load environment
load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyA-d6QJzHg0Ow7CM3bHvMS-ix1DrZn5SUk"

# ---- Utility functions ----
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000
    )
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, just say "answer is not available in the context".
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    context = "\n".join([doc.page_content for doc in docs])
    response = chain(
        {"input_documents": docs, "context": context, "question": user_question},
        return_only_outputs=True,
    )
    return response['output_text']

# ---- Streamlit app ----
def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="🤖", layout="wide")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    st.title("Chat with PDF files using Gemini 🙋‍♂️")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

    if prompt := st.chat_input("Your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"**User:** {prompt}")

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(f"**Assistant:** {response}")

if __name__ == "__main__":
    main()
