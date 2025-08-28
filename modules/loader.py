# Ekstraksi teks dari file
from io import BytesIO
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document as DocxDocument
from pptx import Presentation as PptxPresentation
from pypdf import PdfReader

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    bio = BytesIO(raw)

    if name.endswith(".pdf"):
        reader = PdfReader(bio)
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    elif name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")
    elif name.endswith(".docx"):
        doc = DocxDocument(bio)
        return "\n".join([p.text for p in doc.paragraphs])
    elif name.endswith(".pptx"):
        prs = PptxPresentation(bio)
        return "\n".join([shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")])
    else:
        return ""

def build_documents_from_uploads(uploaded_files):
    docs = []
    for f in uploaded_files:
        text = extract_text_from_file(f)
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"source_file": f.name, "chunk_id": i}))
    return docs
