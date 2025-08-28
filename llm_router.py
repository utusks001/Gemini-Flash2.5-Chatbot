# LLM Dispatcher
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def get_llm_response(question, context, llm_choice):
    prompt_template = PromptTemplate.from_template(
        "Jawablah berdasarkan konteks berikut:\n\n{context}\n\nPertanyaan:\n{question}\n\nJawaban:"
    )
    prompt = prompt_template.format(context=context, question=question)

    if llm_choice == "Gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    elif llm_choice == "GROQ-LLaMA3":
        llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
    elif llm_choice == "GROQ-Mixtral":
        llm = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))
    else:
        raise ValueError("LLM tidak dikenali.")

    return llm.invoke(prompt).content
