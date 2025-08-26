# pdf_handler.py

from PyPDF2 import PdfReader

def extract_text_from_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text
