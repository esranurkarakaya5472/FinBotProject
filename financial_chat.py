import os
import sys
import time
import html
import tempfile
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

class FinancialChat:
    """
    FINBOT Apps - Finansal Raporlarla Sohbet Modülü
    Bu sınıf, finansal PDF raporlarını işleyerek sorulara yanıt veren 
    temel RAG (Retrieval-Augmented Generation) akışını içerir.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            print("UYARI: GOOGLE_API_KEY bulunamadı. Lütfen environment variable olarak tanımlayın veya parametre olarak geçin.")
        self.vector_store = None
        self.retriever = None
        self.chain = None

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

    def initialize_knowledge_base(self, file_path: str):
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
        # Google Generative AI Embeddings kullanımı
        print("Adım 3: Embedding modeli hazırlanıyor (models/gemini-embedding-001)...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=self.api_key)

        # ---------------------------------------------------------
        # ADIM 4: Vektör Veritabanı (Vector Store Indexing)
        # ---------------------------------------------------------
        print("Adım 4: Vektör veritabanına kaydediliyor (ChromaDB)...")
        print("Veri İşleniyor: Lütfen bekleyiniz...")
        time.sleep(15)
        self.vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
        print("Bilgi tabanı hazır!")

    def ask_question(self, query: str):
        """
        Hazırlanmış vektör veritabanını kullanarak soruya cevap verir.
        """
        if not self.vector_store:
            return "Hata: Önce initialize_knowledge_base() fonksiyonu çalıştırılmalı!"

        # ---------------------------------------------------------
        # ADIM 5: Sorgu / Retrieval (Benzerlik Araması)
        # ---------------------------------------------------------
        print(f"\nAdım 5: Kullanıcı sorusu için alakalı içerikler getiriliyor... Soru: {query}")
        self.retriever = self.vector_store.as_retriever()
        
        # ---------------------------------------------------------
        # ADIM 6: Yanıt Üretme (Generation)
        # ---------------------------------------------------------
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
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = rag_chain.invoke(query)
        return answer


def _inject_css(st):
    """Premium FinTech SaaS UI Tasarımı - Glassmorphism & Modern Aesthetics"""
    st.markdown("""
    <style>
        /* ========== GLOBAL & FONT IMPORTS ========== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        
        /* ========== MAIN APP BACKGROUND ========== */
        .stApp {
            background: linear-gradient(180deg, #0a1929 0%, #05070a 40%, #000000 100%) !important;
        }
        
        .main .block-container {
            background: transparent !important;
            padding: 1.5rem 2rem !important;
            max-width: 1400px !important;
        }
        
        /* ========== SIDEBAR: DARK NAVY GLASSMORPHISM ========== */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #05070a 0%, #0a0e1a 100%) !important;
            backdrop-filter: blur(30px) saturate(150%) !important;
            -webkit-backdrop-filter: blur(30px) saturate(150%) !important;
            border-right: 1px solid rgba(0, 212, 255, 0.15) !important;
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.6) !important;
        }
        
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 2rem !important;
        }
        
        /* Hide sidebar collapse button arrows & keyboard text */
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        
        button[kind="header"] {
            display: none !important;
        }
        
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        
        /* ========== LOGO STYLING (HTML/CSS - NO FULLSCREEN BUTTON) ========== */
        .sidebar-logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 1.5rem 0 2rem 0;
            margin-bottom: 1rem;
            border-bottom: 1px solid rgba(0, 212, 255, 0.1);
        }
        
        .sidebar-logo {
            width: 140px;
            height: auto;
            filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.4));
            transition: all 0.3s ease;
        }
        
        .sidebar-logo:hover {
            filter: drop-shadow(0 0 30px rgba(0, 212, 255, 0.7));
            transform: scale(1.05);
        }
        
        /* ========== SIDEBAR ELEMENTS ========== */
        [data-testid="stSidebar"] .stMarkdown {
            color: #e0e7ff !important;
        }
        
        [data-testid="stSidebar"] h3 {
            display: none !important; /* Remove "FinBot" text under logo */
        }
        
        [data-testid="stSidebar"] hr {
            border-color: rgba(0, 212, 255, 0.15) !important;
            margin: 1rem 0 !important;
        }
        
        [data-testid="stSidebar"] .stExpander {
            background: rgba(15, 25, 45, 0.4) !important;
            border: 1px solid rgba(0, 212, 255, 0.1) !important;
            border-radius: 12px !important;
            margin-bottom: 0.75rem !important;
        }
        
        /* ========== WELCOME SCREEN (No File Uploaded) ========== */
        .welcome-card {
            background: linear-gradient(135deg, rgba(15, 25, 45, 0.8) 0%, rgba(10, 20, 40, 0.6) 100%);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 24px;
            padding: 3rem 2.5rem;
            text-align: center;
            margin: 4rem auto;
            max-width: 600px;
            backdrop-filter: blur(20px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            animation: welcomeFadeIn 0.8s ease;
        }
        
        @keyframes welcomeFadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .welcome-card h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FFD700 0%, #00d4ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1rem;
        }
        
        .welcome-card p {
            color: #b0c4de;
            font-size: 1.1rem;
            line-height: 1.6;
            margin-bottom: 0.5rem;
        }
        
        .welcome-icon {
            font-size: 4rem;
            margin-bottom: 1.5rem;
            filter: drop-shadow(0 0 20px rgba(255, 215, 0, 0.5));
        }
        
        /* ========== FILE UPLOAD SUCCESS ========== */
        .upload-success {
            background: linear-gradient(135deg, rgba(0, 200, 120, 0.15) 0%, rgba(0, 255, 136, 0.1) 100%);
            border: 2px solid rgba(0, 255, 136, 0.4);
            border-radius: 16px;
            padding: 1rem 1.5rem;
            margin: 1.5rem 0;
            animation: successSlideIn 0.5s ease;
            font-weight: 600;
            color: #00ff88;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 4px 20px rgba(0, 255, 136, 0.2);
        }
        
        .upload-success .check {
            font-size: 1.5rem;
            animation: checkPulse 0.6s ease;
        }
        
        @keyframes successSlideIn {
            from { opacity: 0; transform: translateX(-30px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes checkPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.2); }
        }
        
        /* ========== MESSAGE BUBBLES: iMessage Style ========== */
        .message-container {
            margin: 1.5rem 0;
            clear: both;
            display: flex;
            gap: 12px;
        }
        
        .message-container.user {
            justify-content: flex-end;
        }
        
        .message-container.bot {
            justify-content: flex-start;
        }
        
        .bot-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            flex-shrink: 0;
            box-shadow: 0 4px 12px rgba(255, 215, 0, 0.3);
        }
        
        .user-message {
            background: linear-gradient(135deg, #0084ff 0%, #0067d6 100%) !important;
            color: #ffffff !important;
            padding: 14px 20px !important;
            border-radius: 20px 20px 4px 20px !important;
            max-width: 65%;
            box-shadow: 0 4px 16px rgba(0, 132, 255, 0.3);
            font-size: 0.95rem;
            line-height: 1.6;
            word-wrap: break-word;
            animation: messageFadeIn 0.3s ease;
        }
        
        .bot-message {
            background: linear-gradient(135deg, rgba(30, 42, 58, 0.9) 0%, rgba(45, 58, 79, 0.85) 100%) !important;
            color: #e8f0fe !important;
            padding: 14px 20px !important;
            border-radius: 20px 20px 20px 4px !important;
            max-width: 65%;
            border: 1px solid rgba(0, 212, 255, 0.15);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            font-size: 0.95rem;
            line-height: 1.65;
            word-wrap: break-word;
            animation: messageFadeIn 0.3s ease;
        }
        
        @keyframes messageFadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .bot-message .bot-label {
            color: #FFD700;
            font-weight: 700;
            font-size: 0.8rem;
            margin-bottom: 8px;
            display: block;
            letter-spacing: 0.5px;
        }
        
        /* ========== CHAT INPUT: NEON BLUE BORDER ========== */
        [data-testid="stChatInput"] {
            position: sticky !important;
            bottom: 20px !important;
            background: rgba(10, 25, 47, 0.95) !important;
            border-radius: 28px !important;
            border: 2px solid rgba(0, 212, 255, 0.6) !important;
            padding: 10px 16px !important;
            backdrop-filter: blur(20px) !important;
            box-shadow: 0 8px 32px rgba(0, 212, 255, 0.25), 
                        0 0 20px rgba(0, 212, 255, 0.15) !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="stChatInput"]:focus-within {
            border-color: rgba(0, 212, 255, 0.9) !important;
            box-shadow: 0 8px 40px rgba(0, 212, 255, 0.4), 
                        0 0 30px rgba(0, 212, 255, 0.3) !important;
        }
        
        [data-testid="stChatInput"] input {
            color: #e0e7ff !important;
            font-size: 0.95rem !important;
        }
        
        [data-testid="stChatInput"] input::placeholder {
            color: rgba(176, 196, 222, 0.5) !important;
        }
        
        /* ========== HEADER TITLE ========== */
        .finbot-title {
            font-weight: 800 !important;
            font-size: 2rem !important;
            margin: 0 !important;
            background: linear-gradient(135deg, #FFD700 0%, #00d4ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.5px;
        }
        
        /* ========== FILE UPLOADER ========== */
        [data-testid="stFileUploader"] {
            background: rgba(15, 25, 45, 0.5);
            border: 2px dashed rgba(0, 212, 255, 0.3);
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(0, 212, 255, 0.6);
            background: rgba(15, 25, 45, 0.7);
        }
        
        /* ========== HIDE STREAMLIT DEFAULTS ========== */
        #MainMenu, header, footer, [data-testid="stToolbar"], 
        .stDeployButton, [data-testid="stDecoration"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }
        
        /* ========== SCROLLBAR ========== */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(10, 25, 47, 0.5);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(0, 212, 255, 0.4);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 212, 255, 0.6);
        }
    </style>
    """, unsafe_allow_html=True)


def run_streamlit_app():
    import streamlit as st
    from pathlib import Path

    st.set_page_config(
        page_title="FinBot Pro - AI Financial Assistant", 
        page_icon="�", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    _inject_css(st)

    API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyAiQQzd9bVAgnXbRo0V0G_EZ1X1_5AVmbA"
    os.environ["GOOGLE_API_KEY"] = API_KEY
    DEFAULT_PDF = "FINBOT Apps – Kurumsal Finans İçin LLM Tabanlı Akıllı Uygulamalar.pdf"
    LOGO_PATH = Path(__file__).resolve().parent / "assets" / "finbot_logo.png"
    FALLBACK_LOGO = "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "file_analyzed" not in st.session_state:
        st.session_state.file_analyzed = False
    if "bot" not in st.session_state:
        st.session_state.bot = None
    if "active_name" not in st.session_state:
        st.session_state.active_name = None

    # ========== SIDEBAR: LOGO WITH HTML/CSS (NO FULLSCREEN ISSUE) ==========
    with st.sidebar:
        # Logo using HTML/CSS instead of st.image to prevent fullscreen button
        logo_url = str(LOGO_PATH) if LOGO_PATH.exists() else FALLBACK_LOGO
        st.markdown(f"""
        <div class="sidebar-logo-container">
            <img src="{logo_url}" class="sidebar-logo" alt="FinBot Logo" 
                 onerror="this.src='{FALLBACK_LOGO}'">
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.expander("📋 Analiz Geçmişi", expanded=False):
            if st.session_state.messages:
                for m in reversed(st.session_state.messages[-10:]):
                    role = "👤 Siz" if m["role"] == "user" else "🤖 FinBot"
                    content = (m["content"][:50] + "...") if len(m["content"]) > 50 else m["content"]
                    st.caption(f"**{role}:** {content}")
            else:
                st.caption("Henüz sohbet geçmişi yok.")
        
        with st.expander("ℹ️ Hakkında", expanded=False):
            st.markdown("""
            **FinBot Pro** yüklediğiniz PDF veya Excel raporlarını analiz eder ve finansal sorularınızı yanıtlar.
            
            🔹 **RAG Teknolojisi** ile güçlendirilmiş  
            🔹 **Gemini AI** entegrasyonu  
            🔹 **Gerçek zamanlı** analiz
            """)

    # ========== FILE UPLOAD ==========
    uploaded_file = st.file_uploader(
        "📄 Finansal Rapor Yükleyin (PDF veya Excel)", 
        type=["pdf", "xlsx", "xls"], 
        key="uploader"
    )
    
    if uploaded_file:
        suffix = ".pdf" if uploaded_file.name.lower().endswith(".pdf") else ".xlsx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            path = tmp.name
        try:
            bot = FinancialChat(api_key=API_KEY)
            with st.spinner("🔍 FinBot Verileri Analiz Ediyor..."):
                bot.initialize_knowledge_base(path)
            st.session_state.bot = bot
            st.session_state.active_name = uploaded_file.name
            st.session_state.file_analyzed = True
        except Exception as e:
            st.error(f"❌ Analiz hatası: {e}")
            st.session_state.file_analyzed = False
    else:
        st.session_state.file_analyzed = False
        if os.path.exists(DEFAULT_PDF):
            if st.session_state.bot is None:
                with st.spinner("🔍 FinBot Verileri Analiz Ediyor..."):
                    bot = FinancialChat(api_key=API_KEY)
                    bot.initialize_knowledge_base(DEFAULT_PDF)
                    st.session_state.bot = bot
                    st.session_state.active_name = "Varsayılan Rapor"

    # ========== SUCCESS MESSAGE ==========
    if st.session_state.file_analyzed and uploaded_file:
        st.markdown("""
        <div class="upload-success">
            <span class="check">✓</span>
            <span>Dosya başarıyla analiz edildi ve hazır!</span>
        </div>
        """, unsafe_allow_html=True)

    # ========== HEADER WITH LOGO & TITLE ==========
    c1, c2 = st.columns([1, 9])
    with c1:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=56)
        else:
            st.image(FALLBACK_LOGO, width=56)
    with c2:
        st.markdown("<p class='finbot-title'>FinBot Pro</p>", unsafe_allow_html=True)
        if st.session_state.active_name:
            st.markdown(
                f"<p style='color:#8892a0; font-size:0.9rem; margin-top:-8px;'>"
                f"📊 Aktif Analiz: <b style='color:#00d4ff;'>{st.session_state.active_name}</b></p>", 
                unsafe_allow_html=True
            )

    # ========== WELCOME SCREEN (NO FILE UPLOADED) ==========
    if not st.session_state.bot and not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">💎</div>
            <h1>FinBot Pro'ya Hoş Geldiniz</h1>
            <p>Finansal raporlarınızı yükleyin ve AI destekli analizlerden faydalanın.</p>
            <p style="font-size:0.9rem; color:#8892a0; margin-top:1rem;">
                👆 Başlamak için yukarıdan bir PDF veya Excel dosyası yükleyin
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ========== CHAT MESSAGES: iMessage Style ==========
    for msg in st.session_state.messages:
        safe = html.escape(msg["content"]).replace("\n", "<br>")
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="message-container user">
                <div class="user-message">{safe}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message-container bot">
                <div class="bot-avatar">🤖</div>
                <div class="bot-message">
                    <span class="bot-label">FINBOT AI</span>
                    {safe}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ========== CHAT INPUT (PINNED TO BOTTOM) ==========
    prompt = st.chat_input("💬 Finansal bir soru sorun... (örn: Gelir tablosu nedir?)")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        bot = st.session_state.get("bot")
        if not bot:
            response = "⚠️ Lütfen önce bir PDF veya Excel dosyası yükleyin."
        else:
            with st.spinner("🔍 FinBot Verileri Analiz Ediyor..."):
                try:
                    response = bot.ask_question(prompt)
                except Exception as e:
                    response = f"❌ Hata: {e}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    if "streamlit" in sys.modules:
        run_streamlit_app()
    else:
        bot = FinancialChat(api_key="AIzaSyAiQQzd9bVAgnXbRo0V0G_EZ1X1_5AVmbA")
        pdf_path = "FINBOT Apps – Kurumsal Finans İçin LLM Tabanlı Akıllı Uygulamalar.pdf"
        if os.path.exists(pdf_path):
            print(f"Bot başlatılıyor... '{pdf_path}' işleniyor...")
            try:
                bot.initialize_knowledge_base(pdf_path)
                print("-" * 50)
                print(f"Bot hazır! '{pdf_path}' üzerinden sorularınızı yanıtlayacak.")
                print("Çıkış yapmak için 'q', 'exit' veya 'çıkış' yazabilirsiniz.\n")
                while True:
                    user_input = input("Soru: ")
                    if user_input.lower() in ['q', 'exit', 'çıkış']:
                        print("Programdan çıkılıyor. İyi günler!")
                        break
                    if user_input.strip() == "":
                        continue
                    try:
                        response = bot.ask_question(user_input)
                        print("-" * 30)
                        print(f"Bot Yanıtı: {response}")
                        print("-" * 30)
                    except Exception as e:
                        print(f"Bir hata oluştu: {e}")
            except Exception as e:
                print(f"Başlangıç hatası: {e}")
        else:
            print(f"Hata: {pdf_path} dosyası bulunamadı.")
