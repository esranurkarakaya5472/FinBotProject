import os
import time
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class LLMClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            try:
                self.api_key = st.secrets["GOOGLE_API_KEY"]
            except Exception:
                pass
                
        if not self.api_key:
            print("UYARI: GOOGLE_API_KEY bulunamadı. Lütfen environment variable veya st.secrets üzerinden tanımlayın.")
            
    def generate_answer(self, query: str, retriever):
        print("Rate Limit (Kota) önlemi: Yanıt üretmeden önce 2 saniye bekleniyor...")
        time.sleep(2)
        print("Adım 6: LLM tarafından yanıt üretiliyor (gemini-flash-latest)...")
        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=self.api_key)
        
        template = """Aşağıdaki bağlamı kullanarak soruyu yanıtlayın.
        Bağlam:
        {context}
        
        Soru: {question}
        
        Yanıt:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = rag_chain.invoke(query)
        return answer
