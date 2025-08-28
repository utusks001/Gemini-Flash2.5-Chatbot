# Streamlit for Gemini Flash 2.5 Chatbot - Upload files and asking

  ### app.py (Conversational Retrieval QA Chatbot)--> https://gemini-flash25-chatbot-uploadpdfs-and-asking.streamlit.app/

  ### streamlit-app.py (Upload a PDF atau TXT file) --> https://gemini-flash25-chatbot-uploadpdf.streamlit.app/

  ### app-streamlit.py (Gemini 2.5 Flash Chatbot — Multi-file (PDF/TXT/DOCX/PPTX)) --> https://gemini-flash25-chatbot-unggahpdf-file.streamlit.app/

  ### main.py (Conversational Retrieval QA (Gemini + FAISS)) --> https://gemini-flash25-chatbot-upload-multifile.streamlit.app/

# 💬 Gemini PDF Chatbot

Chatbot berbasis [Gemini 2.5 Flash](https://makersuite.google.com/) yang memungkinkan pengguna mengunggah file PDF dan mengajukan pertanyaan berdasarkan isi dokumen. Cocok untuk membantu berbagai aplikasi yang berhubungan dengan konteks seperti asisten pemasaran properti, analisis dokumen teknik, analisis dokumen hukum, analisis dokumen sejarah, analisis dokumen menu kuliner, edukasi interaktif, dsb.

## 🚀 Fitur
- Upload dan proses banyak PDF
- Chat dengan isi dokumen menggunakan Gemini 2.5 Flash
- Simpan data kontak pengguna
- Modular dan siap deploy ke Streamlit Cloud 

## 🧱 Struktur Modular
- 'app-streamlit.py' : pilihan UI utama Streamlit
- `.env`: Google API Key
-  Modul fungsional (Parsing File, Text splitter, File extractors, Build documents & FAISS, LangChain, VectorStore, Embeddings, LLM, Chunking)

# Setup & Installation

## 1. Clone atau download repostiroy ini
git clone https://github.com/utusks001/Gemini-Flash2.5-Chatbot.git

## 2. Install Dependencies:
pip install -r requirements.txt
//Make sure you have Python 3.7+ installed.

## 3. Set Environment Variables: buat file dengan nama .env kemudian isi file tersebut dengan API Keys sebagai berikut Example:
- GROQ_API_KEY = "GROQQ API KEY"
- GOOGLE_API_KEY = "GOOGLE API KEY"

## 4. Run Streamlit di local
streamlit run main.py

## 5. Lalu deploy ke https://share.streamlit.io/
