import os
import time
import streamlit as st
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class EmbeddingManager:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            try:
                self.api_key = st.secrets["GOOGLE_API_KEY"]
            except Exception:
                pass

    def _load_documents(self, file_path: str) -> List[Document]:
        """PDF veya Excel dosyasından belgeleri yükler."""
        low = file_path.lower()
        if low.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            return loader.load()
        if low.endswith((".xlsx", ".xls")):
            import pandas as pd
            documents = []
            xl = pd.ExcelFile(file_path)
            for sheet_name in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=sheet_name)
                text = df.to_string(index=False)
                if text.strip():
                    documents.append(Document(page_content=text, metadata={"source": file_path, "sheet": sheet_name}))
            return documents
        raise ValueError(f"Desteklenmeyen dosya türü: {file_path}")

    def create_vector_store(self, file_path: str):
        """
        PDF veya Excel dosyasını yükler, parçalar ve vektör veritabanını oluşturur.
        Bu işlem bot başlatıldığında sadece bir kez yapılır.
        """
        # ---------------------------------------------------------
        # ADIM 1: Veri Alma (Data Ingestion)
        # ---------------------------------------------------------
        print(f"Adım 1: '{file_path}' dosyası yükleniyor...")
        documents = self._load_documents(file_path)
        print(f"   -> {len(documents)} sayfa/sheet yüklendi.")

        # ---------------------------------------------------------
        # ADIM 2: Embedding (Metinsel Parçalama / Chunking)
        # ---------------------------------------------------------
        print("Adım 2: Metin parçalara ayrılıyor (Chunking)...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        print(f"   -> {len(splits)} parça oluşturuldu.")

        # ---------------------------------------------------------
        # ADIM 3: Embedding Oluşturma (Vectorization)
        # ---------------------------------------------------------
        print("Adım 3: Embedding modeli hazırlanıyor (models/gemini-embedding-001)...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=self.api_key)

        # ---------------------------------------------------------
        # ADIM 4: Vektör Veritabanı (Vector Store Indexing)
        # ---------------------------------------------------------
        print("Adım 4: Vektör veritabanına kaydediliyor (ChromaDB)...")
        print("Veri İşleniyor: Lütfen bekleyiniz...")
        time.sleep(15)
        vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
        print("Bilgi tabanı hazır!")
        return vector_store
