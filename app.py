# -*- coding: utf-8 -*-
# Uygulama: Focusia (AI Focus Mentor)
# Amaç: Kullanıcıdan kısa bir durum alıp, RAG (Chroma + MMR) bağlamıyla
#       HuggingFace LLM'e vererek 3 öneri + 1 mini egzersiz üretmek.

import os
import streamlit as st
from data_loader import build_or_load_chroma   # Chroma DB kurulum/yükleme
from rag_pipeline import make_retrieval_chain  # LLM + retriever + prompt zinciri

# ====== MARKA ======
# Ortam değişkenlerinden okunur; yoksa varsayılanları kullan.
PRODUCT_NAME = os.getenv("PRODUCT_NAME", "Focusia")
TAGLINE = os.getenv("TAGLINE", "Yapay zeka destekli odak koçun.")
LOGO_PATH = os.getenv("LOGO_PATH", "logo.png")     
FAVICON_PATH = os.getenv("FAVICON_PATH", "assets/favicon.png")
# ====================

# Sayfa başlığı ve favicon ayarı
page_icon = FAVICON_PATH if os.path.exists(FAVICON_PATH) else "🧠"
st.set_page_config(page_title=PRODUCT_NAME, page_icon=page_icon, layout="centered")

# ====== STİL ======
# Uygulamanın tamamına uygulanan CSS: arka plan, kartlar, butonlar, expander görünümleri.
st.markdown("""
<style>
/* Tam ekran arka plan: lacivert → mor gradient */
[data-testid="stAppViewContainer"]{
  background: linear-gradient(135deg, #1E1B4B 0%, #3730A3 40%, #6D28D9 100%);
  color:#F9FAFB;
}
/* Sidebar (ileride gerekirse) */
[data-testid="stSidebar"]{
  background: linear-gradient(135deg, #312E81 0%, #5B21B6 100%);
}
/* İç blok/kart etkisi: içerik alanını kart gibi göster */
.block-container{
  background-color: rgba(255,255,255,0.05);
  padding: 2rem 3rem;
  border-radius: 18px;
  box-shadow: 0 4px 14px rgba(0,0,0,0.35);
}
/* Başlık (logo + ürün adı + tagline) */
.app-head{display:flex; align-items:center; gap:18px; margin-bottom:8px;}
.app-logo{border-radius:20px; width:110px; height:110px; object-fit:contain;}
.app-title{font-size:2rem; font-weight:700; margin:0; color:#E0E7FF;}
.app-tag{color:#C7D2FE; margin-top:4px; font-size:1rem;}
/* Form alanları: koyu zemin + okunabilir metin */
label[for]{ color:#EDE9FE !important; font-weight:600; }
textarea{
  background-color: rgba(30,27,75,0.85) !important; /* koyu lacivert */
  color:#E0E7FF !important;
  border-radius: 12px !important;
  border:1px solid rgba(255,255,255,0.2) !important;
}
textarea::placeholder{ color:#C7D2FE !important; opacity:0.9 !important; }
/* Metinler & ayraçlar */
hr{ border-top:1px solid rgba(255,255,255,0.2); }
p, .stMarkdown{ color:#EDE9FE !important; }
/* 💡 Öneri Kartı: yarı saydam "glass" görünüm */
.focusia-card{
  background: linear-gradient(145deg, rgba(67,56,202,0.25), rgba(124,58,237,0.25));
  border:1px solid rgba(199,210,254,0.35);
  border-radius:14px;
  padding:1.4rem 1.6rem;
  margin-top:1rem;
  color:#F3E8FF;
  box-shadow: 0 4px 14px rgba(67,56,202,0.3);
  backdrop-filter: blur(12px);
}
.focusia-card h3{ color:#E0E7FF; margin-bottom:0.8rem; }
/* Kaynak listesi görünümü */
.focusia-source{
  margin-top:1rem; padding-top:0.5rem;
  border-top:1px solid rgba(255,255,255,0.2);
  color:#C7D2FE; font-size:0.9rem;
}
/* Buton: her durumda gradient; hover'da koyulaşır */
[data-testid="stForm"] button,
.stForm button,
div.stButton > button,
button[kind="primary"],
button[kind="secondary"],
button[data-testid="baseButton-primary"],
button[data-testid="baseButton-secondary"] {
  background-image: linear-gradient(90deg, #4338CA, #7C3AED) !important;
  background-color: transparent !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 8px rgba(67,56,202,0.35) !important;
  transition: all 0.25s ease-in-out !important;
}
[data-testid="stForm"] button:hover,
.stForm button:hover,
div.stButton > button:hover,
button[data-testid="baseButton-primary"]:hover,
button[data-testid="baseButton-secondary"]:hover {
  background-image: linear-gradient(90deg, #5B21B6, #4F46E5) !important;
  box-shadow: 0 3px 12px rgba(67,56,202,0.55) !important;
  transform: translateY(-1px);
}
[data-testid="stForm"] button:active,
.stForm button:active,
div.stButton > button:active {
  transform: translateY(0);
  box-shadow: 0 1px 6px rgba(67,56,202,0.45) !important;
}
[data-testid="stForm"] button:disabled,
.stForm button:disabled {
  background-image: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
  opacity: .7 !important;
  cursor: not-allowed !important;
}
/* Expander (Kaynaklar) teması */
[data-testid="stExpander"]{
  background: rgba(30,27,75,0.5);
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.2);
}
[data-testid="stExpander"] div[role="button"]{
  color:#C7D2FE !important; font-weight:500;
}
</style>
""", unsafe_allow_html=True)
# ====== /STİL ======

# ====== BAŞLIK ======
# Logo + ürün adı + tagline (yukarıdaki CSS'le şekilleniyor)
st.markdown("<div class='app-wrap'>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 6])
with col1:
    # LOGO_PATH gerçek dosyayı işaret ediyorsa resmi göster, yoksa emoji fallback
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=110)
    else:
        st.markdown("<div class='app-logo'>🧠</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='app-head'><h1 class='app-title'>{PRODUCT_NAME}</h1></div>", unsafe_allow_html=True)
    st.markdown(f"<p class='app-tag'>{TAGLINE}</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.divider()
# ====== /BAŞLIK ======

@st.cache_resource(show_spinner=False)
def _get_chain():
    """
    Chroma veritabanını hazırla ve MMR tabanlı bir retriever ile
    LLM zincirini (prompt+LLM) kur. Dönen obje, .invoke({'query': ...})
    arayüzünü destekler.
    """
    db = build_or_load_chroma()
    # MMR (Maximal Marginal Relevance): tekrar eden benzer sonuçları azaltır.
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    )
    return make_retrieval_chain(retriever)

# Zinciri bir kez oluştur (cache_resource sayfada tekrar çalıştırmayı önler)
qa = _get_chain()

# ====== FORM ======
# Kullanıcıdan kısa bir durum yazması istenir.
with st.form("focus_form"):
    user_input = st.text_area(
        "Bugün nasılsın? (odak durumunu kısaca anlat)",
        placeholder="Örn: Sürekli telefona bakıyorum, başladığım işi bitiremiyorum..."
    )
    submitted = st.form_submit_button("Öneri al")
# ====== /FORM ======

# ====== SORGULAMA & YANIT ======
if submitted:
    if not user_input.strip():
        # Boş girişe uyarı göster
        st.warning("Lütfen kısa bir durum cümlesi yaz.")
    else:
        # LLM cevabını beklerken spinner göster
        with st.spinner("Önerin hazırlanıyor..."):
            try:
                # RAG zincirini çağır: {'result': ..., 'source_documents': ...}
                result = qa.invoke({"query": user_input})
            except Exception as e:
                # Herhangi bir hata kullanıcıya görünür olsun
                st.error(f"Hata: {e}")
                st.stop()

        # Modelden dönen nihai öneri metni
        answer = result.get("result", result)

        # 💡 Öneri Kartı (temaya uygun şık kart)
        st.markdown("<div class='focusia-card'><h3>Senin İçin Öneriler</h3>", unsafe_allow_html=True)
        st.markdown(answer, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Kaynaklar (RAG dayanakları): kullanıcı isterse açıp görebilsin
        srcs = result.get("source_documents", [])
        if srcs:
            with st.expander("Kaynakları Gör (RAG dayanakları)"):
                # İsteğe bağlı: ham pasajları da göster (kanıt)
                show_snippets = st.checkbox("Ham pasajları da göster", value=False)

                st.markdown(
                    "<div class='focusia-source'><b>Kullanılan Teknik ve Kaynaklar:</b>",
                    unsafe_allow_html=True
                )

                # Aynı başlığı tekrar etmemek için tekilleştir
                seen = set()
                for d in srcs:
                    topic = (d.metadata.get("topic", "") or "focus_tips.json satırı").strip()
                    if topic in seen:
                        continue
                    seen.add(topic)

                    # Kaynak başlığı
                    st.markdown(f"- **{topic}**")

                    # İstenirse kısa bir ham pasaj (kanıt niteliğinde)
                    if show_snippets:
                        txt = getattr(d, "page_content", "")
                        snippet = (txt[:220] + "…") if len(txt) > 220 else txt
                        st.code(snippet, language="markdown")

                st.markdown("</div>", unsafe_allow_html=True)
# ====== /SORGULAMA & YANIT ======

