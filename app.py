import streamlit as st
import os
import tempfile
from embedding import EmbeddingManager
from retriever import DocumentRetriever
from llm_client import LLMClient

# ─────────────────────────────────────────────────────────────────
# 1. API KONFİGÜRASYONU
# ─────────────────────────────────────────────────────────────────
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    try:
        API_KEY = st.secrets["GOOGLE_API_KEY"]
    except:
        API_KEY = "AIzaSyAiQQzd9bVAgnXbRo0V0G_EZ1X1_5AVmbA"
os.environ["GOOGLE_API_KEY"] = API_KEY

# Sayfa Ayarları
st.set_page_config(
    page_title="FinBot Pro | AI Intelligence",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# 2. RICH RADIAL GRADIENT & GLOW EFFECTS CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg-deep: #020617;
        --bg-sidebar: #020617;
        --accent-blue: #3b82f6;
        --accent-purple: #818cf8;
        --glass-bg: rgba(255, 255, 255, 0.02);
        --glass-border: rgba(255, 255, 255, 0.06);
        --text-main: #f1f5f9;
        --glow-blue: rgba(59, 130, 246, 0.5);
    }

    /* VIBRANT RADIAL GRADIENT BACKGROUND */
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e293b 0%, #020617 70%) !important;
        color: var(--text-main) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--glass-border) !important;
        box-shadow: 10px 0 50px rgba(0, 0, 0, 0.8);
    }

    /* GLOWING HEADER */
    .sidebar-header {
        padding: 1.5rem 0;
        text-align: center;
        border-bottom: 1px solid var(--glass-border);
        margin-bottom: 2rem;
    }
    .sidebar-header h1 {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        /* Parlama Efekti */
        filter: drop-shadow(0 0 15px rgba(59, 130, 246, 0.4));
        transition: all 0.3s ease;
    }
    .sidebar-header h1:hover {
        filter: drop-shadow(0 0 25px rgba(59, 130, 246, 0.8));
        transform: scale(1.02);
    }

    /* GLOWING BUTTONS */
    .stButton > button {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        color: white !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 20px var(--glow-blue) !important;
        transform: translateY(-2px);
    }

    [data-testid="stFileUploader"] {
        background: var(--glass-bg);
        border: 1px dashed var(--glass-border);
        border-radius: 12px;
        padding: 5px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-blue);
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.2);
    }

    .chat-container {
        max-width: 950px;
        margin: 0 auto;
        padding-bottom: 200px !important; /* Streamlit boşluklarını zorla ezer */
        position: relative;
    }

    .message-wrapper {
        display: flex;
        flex-direction: column;
        margin-bottom: 1.5rem;
        animation: slideUp 0.4s ease-out;
    }
    .user-wrapper { align-items: flex-end; }
    .bot-wrapper { align-items: flex-start; }

    .msg-bubble {
        padding: 0.9rem 1.4rem;
        border-radius: 22px;
        font-size: 0.98rem;
        line-height: 1.6;
        max-width: 80%;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
        height: auto !important;
        min-height: fit-content;
        word-wrap: break-word;
        margin-bottom: 40px !important; /* Oval yapıyı korur ve mesafe sağlar */
    }
    .msg-bubble:hover {
        transform: scale(1.005);
    }

    .user-msg {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        border-bottom-right-radius: 4px;
        box-shadow: 0 5px 20px rgba(37, 99, 235, 0.3);
    }

    .bot-msg {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(20px);
        color: #e2e8f0;
        border-bottom-left-radius: 4px;
    }

    .hero-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 65vh;
        text-align: center;
    }
    .hero-card {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid var(--glass-border);
        padding: 4rem;
        border-radius: 32px;
        backdrop-filter: blur(40px);
        max-width: 650px;
        box-shadow: 0 40px 100px rgba(0, 0, 0, 0.5);
    }
    .hero-card h2 {
        font-size: 3rem; font-weight: 800; margin-bottom: 1.5rem;
        background: linear-gradient(to bottom, #ffffff, #94a3b8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        /* Kahraman Başlığı Parlama */
        filter: drop-shadow(0 0 20px rgba(255, 255, 255, 0.2));
    }

    div[data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 35px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: 85% !important;
        max-width: 850px !important;
        background: #020617 !important; /* Arkaplan tamamen opak yapıldı */
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 24px !important;
        padding: 12px 20px !important;
        box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.7) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        z-index: 99999 !important; /* En üst katman */
    }

    div[data-testid="stChatInput"] textarea {
        font-size: 1.05rem !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
        line-height: 1.5 !important;
        background: transparent !important;
        border: none !important;
        color: #f8fafc !important;
    }

    div[data-testid="stChatInput"]:focus-within {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 40px rgba(59, 130, 246, 0.3), 0 25px 60px -15px rgba(0, 0, 0, 0.7) !important;
        transform: translateX(-50%) translateY(-2px) !important;
    }

    /* Input yanındaki gönder butonu için stil */
    div[data-testid="stChatInput"] button {
        background: var(--accent-blue) !important;
        border-radius: 12px !important;
        padding: 8px !important;
        bottom: 12px !important;
        right: 15px !important;
    }

    @keyframes pulse {
        0% { transform: scale(0.8); opacity: 0.5; box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7); }
        70% { transform: scale(1.2); opacity: 1; box-shadow: 0 0 0 15px rgba(59, 130, 246, 0); }
        100% { transform: scale(0.8); opacity: 0.5; box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
    }
    .loader-text {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# 3. VERİ VE SESSION STATE
# ─────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "is_ready" not in st.session_state:
    st.session_state.is_ready = False

# ─────────────────────────────────────────────────────────────────
# 4. SIDEBAR (GLOWING ELEMENTS)
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header"><h1>💎 FinBot Pro</h1></div>', unsafe_allow_html=True)
    
    st.markdown("### 📄 Rapor Yükleme")
    uploaded_file = st.file_uploader("", type=["pdf", "xlsx", "xls"])
    
    if uploaded_file:
        if not st.session_state.is_ready:
            with st.spinner("🔍 FinBot Verileri Analiz Ediyor... Lütfen bekleyin"):
                suffix = ".pdf" if uploaded_file.name.lower().endswith(".pdf") else ".xlsx"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    path = tmp.name
                
                try:
                    em = EmbeddingManager(api_key=API_KEY)
                    vs = em.create_vector_store(path)
                    st.session_state.vector_store = vs
                    st.session_state.is_ready = True
                    st.success("✅ Hazır")
                    st.rerun()
                except Exception as e:
                    st.error(f"⚠️ Hata: {str(e)}")
        else:
            st.success("✅ Aktif")
            if st.button("🗑️ Belleği Temizle"):
                st.session_state.vector_store = None
                st.session_state.is_ready = False
                st.session_state.messages = []
                st.rerun()
    else:
        st.info("💡 PDF veya Excel dökümanı bekliyor...")
        st.session_state.vector_store = None
        st.session_state.is_ready = False

    st.markdown("---")
    st.caption("v2.9 | Ultra-Vibrant Glow")

# ─────────────────────────────────────────────────────────────────
# 5. ANA EKRAN VE AKILLI AI AKIŞI
# ─────────────────────────────────────────────────────────────────
if not st.session_state.is_ready:
    st.markdown("""
        <div class="hero-section">
            <div class="hero-card">
                <h2>Finansal Zeka.</h2>
                <p style="color: #94a3b8; font-size: 1.2rem; margin-bottom: 2.5rem;">
                    Dökümanlarınızı yükleyin ve derin verilere saniyeler içinde ulaşın.
                </p>
                <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); padding: 12px; border-radius: 12px; color: #60a5fa; font-size: 0.95rem;">
                   Başlamak için sol menüden bir PDF dökümanı yükleyin.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        wrapper_class = "user-wrapper" if msg["role"] == "user" else "bot-wrapper"
        bubble_class = "user-msg" if msg["role"] == "user" else "bot-msg"
        st.markdown(f'<div class="message-wrapper {wrapper_class}"><div class="msg-bubble {bubble_class}">{msg["content"]}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if user_input := st.chat_input("Döküman hakkında bir şeyler sor..."):
        # 1. Mesajı Session State'e ekle
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 2. Kullanıcının mesajını ANINDA ekrana yansıt (Manuel Render)
        st.markdown(f'<div class="message-wrapper user-wrapper"><div class="msg-bubble user-msg">{user_input}</div></div>', unsafe_allow_html=True)
        
        # 3. Yükleme Spinner'ını göster
        loader_placeholder = st.empty()
        with loader_placeholder:
            st.spinner("🔍 FinBot Verileri Analiz Ediyor...")

        try:
            # 4. Yanıtı al
            retriever_mgr = DocumentRetriever(vector_store=st.session_state.vector_store)
            retriever = retriever_mgr.get_retriever(user_input)
            
            llm_client = LLMClient(api_key=API_KEY)
            full_response = llm_client.generate_answer(user_input, retriever)
            
            # Spinner'ı temizle
            loader_placeholder.empty()
            bot_placeholder = st.empty()
            bot_placeholder.markdown(f'<div class="message-wrapper bot-wrapper"><div class="msg-bubble bot-msg">{full_response}</div></div>', unsafe_allow_html=True)
            
            # 6. Yanıtı kalıcı hafızaya al ve sayfayı yenile
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            # (Render edildiği için anında rerun yerine bir sonraki işlemde güncellenir veya istenirse st.rerun() açılabilir)
        except Exception as e:
            st.error(f"❌ Sistem Hatası: {str(e)}")
