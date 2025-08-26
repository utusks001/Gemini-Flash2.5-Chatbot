# Streamlit for Gemini Flash 2.5 Chatbot - Upload files and asking

  ## app.py (Upload Multi-file (PDF+TXT))--> https://gemini-flash25-chatbot-uploadpdfs-and-asking.streamlit.app/

  ## app-streamlit (Upload a PDF atau TXT file) https://gemini-flash25-chatbot-uploadpdf.streamlit.app/

  ## app-streamlit (Upload Multi-file (PDF/TXT/DOCX/PPTX)) --> https://gemini-flash25-chatbot-unggahpdf-file.streamlit.app/

# 💬 Gemini PDF Chatbot

Chatbot berbasis [Gemini 2.5 Flash](https://makersuite.google.com/) yang memungkinkan pengguna mengunggah file PDF dan mengajukan pertanyaan berdasarkan isi dokumen. Cocok untuk aplikasi seperti asisten pemasaran properti, analisis dokumen hukum, atau edukasi interaktif.

## 🚀 Fitur
- Upload dan proses banyak PDF
- Chat dengan isi dokumen menggunakan Gemini 2.5 Flash
- Simpan data kontak pengguna
- Modular dan siap deploy ke Streamlit Cloud 

## 🧱 Struktur Modular
- `app.py`: UI utama Streamlit
- `settings.py`: Konfigurasi API key
-  Modul fungsional (PDF, chunking, vector store, chain, user data)

## 🔧 Instalasi Lokal

```bash
git clone https://github.com/your-username/gemini-pdf-chatbot.git
cd gemini-pdf-chatbot
python -m venv venv
source venv/bin/activate  # atau venv\Scripts\activate di Windows
pip install -r requirements.txt
