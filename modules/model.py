# modules/model.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI

def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
