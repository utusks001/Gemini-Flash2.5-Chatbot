# Pilih Model LLM (Gemini, GROQ, LLaMA)
# Load Gemini, GROQ, atau LLaMA
# modules/model.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

def load_llm(provider="gemini"):
    if provider == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    elif provider == "groq":
        return ChatGroq(model="mixtral-8x7b", api_key=os.getenv("GROQ_API_KEY"))
    elif provider == "llama":
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        return HuggingFacePipeline(pipeline=pipe)
