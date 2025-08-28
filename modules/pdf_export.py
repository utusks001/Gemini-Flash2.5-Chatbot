# Export Jawaban ke PDF
# modules/pdf_export.py
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO

def export_answer_to_pdf(answer: str) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    text_object = c.beginText(40, A4[1] - 50)
    text_object.setFont("Helvetica", 12)
    for line in answer.split("\n"):
        text_object.textLine(line)
    c.drawText(text_object)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer
