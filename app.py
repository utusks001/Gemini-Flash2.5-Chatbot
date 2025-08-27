# app.py
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Fallback imports
from docx import Document as DocxDocument
from pptx import Presentation
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# FAISS index dir
INDEX_DIR = "faiss_index"
DEFAULT_K = 5

st.set_page_config(page_title="📚 Multi-File RAG Chatbot", layout="wide")

# API Key
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Please set GOOGLE_API_KEY in Streamlit secrets.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- Loaders ---
def load_docs(files):
    docs = []
    for f in files:
        file_path = f.name
        with open(file_path, "wb") as out_file:
            out_file.write(f.read())

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())

        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())

        elif file_path.endswith(".docx"):
            try:
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                loader = UnstructuredWordDocumentLoader(file_path)
                docs.extend(loader.load())
            except ImportError:
                # fallback
                docx = DocxDocument(file_path)
                text = "\n".join([p.text for p in docx.paragraphs if p.text.strip()])
                docs.append(Document(page_content=text, metadata={"source_file": file_path}))

        elif file_path.endswith(".pptx"):
            try:
                from langchain_community.document_loaders import UnstructuredPowerPointLoader
                loader = UnstructuredPowerPointLoader(file_path)
                docs.extend(loader.load())
            except ImportError:
                # fallback
                prs = Presentation(file_path)
                text_runs = []
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text.strip():
                            text_runs.append(shape.text)
                text = "\n".join(text_runs)
                docs.append(Document(page_content=text, metadata={"source_file": file_path}))

        else:
            st.warning(f"Unsupported file: {file_path}")

    return docs

# --- FAISS ---
def build_faiss_from_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(chunks, embeddings)

def save_index(vs):
    vs.save_local(INDEX_DIR)

def load_index():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# --- Main ---
def main():
    st.title("📚 Conversational Retrieval QA Chatbot")
    st.write("Upload PDF, TXT, DOCX, PPTX → Index → Ask with Gemini")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Try load saved index
    if os.path.exists(INDEX_DIR):
        try:
            st.session_state.vector_store = load_index()
            st.sidebar.success("Loaded FAISS index from disk.")
        except Exception as e:
            st.sidebar.error(f"Could not load FAISS index: {e}")

    uploaded_files = st.sidebar.file_uploader(
        "Upload files", type=["pdf", "txt", "docx", "pptx"], accept_multiple_files=True
    )

    if st.sidebar.button("Build Index"):
        if uploaded_files:
            with st.spinner("Building FAISS index..."):
                docs = load_docs(uploaded_files)
                vs = build_faiss_from_docs(docs)
                st.session_state.vector_store = vs
                save_index(vs)
                st.sidebar.success("Index built & saved!")
        else:
            st.sidebar.warning("Please upload some files first.")

    query = st.text_input("Ask a question based on indexed documents:")
    k = st.number_input("Number of retrieved chunks (k)", min_value=1, max_value=20, value=DEFAULT_K)

    if st.button("Ask") and query.strip():
        if st.session_state.vector_store is None:
            st.warning("No vector store available. Build or load FAISS first.")
        else:
            with st.spinner("Retrieving..."):
                hits = st.session_state.vector_store.similarity_search(query, k=k)

                if not hits:
                    st.info("No relevant chunks found.")
                else:
                    context_block = "\n\n---\n\n".join(
                        [f"[{i+1}] ({d.metadata.get('source_file','-')}#chunk-{d.metadata.get('chunk_id','-')})\n{d.page_content}"
                         for i,d in enumerate(hits)]
                    )
                    system_instructions = (
                        "Jawablah seakurat mungkin berdasarkan konteks berikut. "
                        "Jika jawaban tidak ada, katakan 'Jawaban tidak tersedia dalam konteks yang ditanyakan, "
                        "pelajari dan berikan tambahan informasi yang diperlukan untuk memperkuat insightfull konteks "
                        "atau topik yang ditanyakan dari berbagai sumber atau referensi dengan menyertakan link website..'"
                    )
                    composed_prompt = f"{system_instructions}\n\n=== KONTEX ===\n{context_block}\n\n=== PERTANYAAN ===\n{query}\n\n=== JAWABAN ===\n\n=== REFERENSI/SUMBER ==="

                    try:
                        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
                        with st.spinner("Generating answer with Gemini..."):
                            resp = llm.invoke(composed_prompt)

                        out_text = getattr(resp, "content", None) or (
                            resp.candidates[0].content if getattr(resp, "candidates", None) else str(resp)
                        )
                        st.subheader("💬 Answer")
                        st.write(out_text)

                        with st.expander("🔎 Sources"):
                            for i,d in enumerate(hits, start=1):
                                st.markdown(f"**[{i}]** {d.metadata.get('source_file','-')} (chunk {d.metadata.get('chunk_id','-')})")
                                st.caption(d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""))

                    except Exception as e:
                        st.error(f"Error calling Gemini: {e}")

if __name__ == "__main__":
    main()

