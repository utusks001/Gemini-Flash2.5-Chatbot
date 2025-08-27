# main.py
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
from PIL import Image
import pytesseract
import pickle

# ==== CONFIG ====
INDEX_DIR = "faiss_index"
DEFAULT_K = 4

# ==== LOAD DOCS ====
def load_docs(uploaded_files):
    docs = []
    for f in uploaded_files:
        ext = os.path.splitext(f.name)[1].lower()
        if ext == ".pdf":
            pdf_reader = PdfReader(f)
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source_file": f.name, "page": i+1}))
        elif ext == ".txt":
            text = f.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source_file": f.name}))
        elif ext == ".docx":
            doc = DocxDocument(f)
            full_text = "\n".join([p.text for p in doc.paragraphs])
            docs.append(Document(page_content=full_text, metadata={"source_file": f.name}))
        elif ext == ".pptx":
            prs = Presentation(f)
            text_runs = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text_runs.append(shape.text)
            docs.append(Document(page_content="\n".join(text_runs), metadata={"source_file": f.name}))
        elif ext in [".png", ".jpg", ".jpeg"]:
            try:
                img = Image.open(f)
                text = pytesseract.image_to_string(img)
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source_file": f.name}))
            except Exception as e:
                st.warning(f"OCR failed for {f.name}: {e}")
        else:
            st.warning(f"Unsupported file type: {f.name}")
    return docs

# ==== BUILD VECTOR STORE ====
def build_faiss_from_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(split_docs, embeddings)

def save_faiss(vs, path=INDEX_DIR):
    vs.save_local(path)

def load_faiss(path=INDEX_DIR):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

# ==== STREAMLIT APP ====
def main():
    st.set_page_config(page_title="Conversational Retrieval QA", layout="wide")
    st.title("📄 Conversational Retrieval QA (Gemini + FAISS)")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Upload docs
    uploaded_files = st.file_uploader("Upload PDF, TXT, DOCX, PPTX, or Image (PNG/JPG)", 
                                      type=["pdf", "txt", "docx", "pptx", "png", "jpg", "jpeg"], 
                                      accept_multiple_files=True)

    if uploaded_files:
        if st.button("📑 Build Index from Uploads"):
            with st.spinner("Indexing documents..."):
                docs = load_docs(uploaded_files)
                if docs:
                    vs = build_faiss_from_docs(docs)
                    save_faiss(vs)
                    st.session_state.vector_store = vs
                    st.success("✅ Index built & saved!")
                else:
                    st.warning("No text extracted from files.")

    # Load existing index
    if st.session_state.vector_store is None and os.path.exists(INDEX_DIR):
        try:
            st.session_state.vector_store = load_faiss()
            st.info("Loaded existing FAISS index from disk.")
        except Exception as e:
            st.error(f"Could not load FAISS index: {e}")

    # Query
    query = st.text_input("💬 Ask a question:")
    k = st.number_input("Number of retrieved chunks (k)", min_value=1, max_value=20, value=DEFAULT_K)

    if st.button("Ask"):
        if st.session_state.vector_store is None:
            st.warning("No vector store available. Upload & build FAISS first.")
        else:
            with st.spinner("Retrieving relevant chunks..."):
                hits = st.session_state.vector_store.similarity_search(query, k=k)
                if not hits:
                    st.info("No relevant chunks found.")
                else:
                    context_block = "\n\n---\n\n".join(
                        [f"[{i+1}] ({d.metadata.get('source_file','-')})\n{d.page_content}" for i,d in enumerate(hits)]
                    )
                    system_instructions = (
                        "Jawablah seakurat mungkin berdasarkan konteks berikut. "
                        "Jika jawaban tidak ada, katakan: 'Jawaban tidak tersedia dalam konteks yang ditanyakan, "
                        "pelajari dan berikan tambahan informasi dari referensi terpercaya dengan menyertakan link website.'"
                    )
                    composed_prompt = f"{system_instructions}\n\n=== KONTEX ===\n{context_block}\n\n=== PERTANYAAN ===\n{query}\n\n=== JAWABAN ==="

                    try:
                        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
                        with st.spinner("Generating answer with Gemini..."):
                            resp = llm.invoke(composed_prompt)

                        # === FIX HANDLING RESPONSE ===
                        if isinstance(resp, str):
                            out_text = resp
                        elif hasattr(resp, "content"):
                            out_text = resp.content
                        elif hasattr(resp, "candidates") and resp.candidates:
                            out_text = resp.candidates[0].content
                        else:
                            out_text = str(resp)

                        st.subheader("💡 Answer")
                        st.write(out_text)

                        with st.expander("📚 Sources"):
                            for i, d in enumerate(hits, start=1):
                                st.markdown(f"**[{i}]** {d.metadata.get('source_file','-')}")
                                st.caption(d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""))

                    except Exception as e:
                        st.error(f"Error calling Gemini: {e}")


if __name__ == "__main__":
    main()
