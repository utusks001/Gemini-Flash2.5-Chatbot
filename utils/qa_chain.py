from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from settings import GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

def build_qa_chain():
    prompt_template = """
    Jawablah pertanyaan berdasarkan konteks dari brosur properti berikut.
    Jika jawaban tidak tersedia, katakan "Informasi tidak tersedia dalam dokumen."

    Konteks:
    {context}

    Pertanyaan:
    {question}

    Jawaban:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
