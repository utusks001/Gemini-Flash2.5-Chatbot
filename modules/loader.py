# Load & split dokumen + OCR image
# OCR dengan EasyOCR
# modules/loader.py

from io import BytesIO
from PIL import Image
import easyocr
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document as DocxDocument
from pptx import Presentation as PptxPresentation
from PyPDF2 import PdfReader

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
ocr_reader = easyocr.Reader(['en', 'id'], gpu=False)

def extract_text_from_image(file_bytes: BytesIO) -> str:
    image = Image.open(file_bytes)
    result = ocr_reader.readtext(image)
    return "\n".join([text[1] for text in result])

def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    bio = BytesIO(raw)

    if name.endswith(".pdf"):
        text = ""
        reader = PdfReader(bio)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    elif name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")
    elif name.endswith(".docx"):
        doc = DocxDocument(bio)
        return "\n".join([p.text for p in doc.paragraphs])
    elif name.endswith(".pptx"):
        prs = PptxPresentation(bio)
        return "\n".join([shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")])
    elif name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".jfif")):
        return extract_text_from_image(bio)
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

def preview_image_and_ocr(uploaded_file):
    image = Image.open(uploaded_file)
    result = ocr_reader.readtext(image)
    text = "\n".join([text[1] for text in result])
    return image, text
