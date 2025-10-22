import streamlit as st
import os
import json
import re
from dotenv import load_dotenv
# KRİTİK DÜZELTME: 'client' yerine 'global_collection' alınıyor
from rag_module import rag_answer_gemini, global_collection 
import textwrap

# --- API Anahtarı Kontrolü ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Tema Durumu ---
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

def toggle_theme():
    """Tema geçişini sağlar (light <-> dark)"""
    st.session_state.theme_mode = "dark" if st.session_state.theme_mode == "light" else "light"

# --- CSS Yükleme ---
def load_css(theme="light"):
    """styles/style.css dosyasını okur ve Streamlit'e enjekte eder."""
    css_file = "styles/style.css"
    try:
        with open(css_file, "r", encoding="utf-8") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ CSS dosyası bulunamadı!")

# --- JS Enjeksiyonu ---
def apply_theme_js(theme):
    """Python'daki 'theme' durumunu HTML body etiketine sınıf olarak ekler."""
    js = f"""
        <script>
            const body = window.parent.document.body;
            body.classList.remove('light', 'dark');
            body.classList.add('{theme}');
        </script>
    """
    st.components.v1.html(js, height=0)

# --- Sabit Uyarı Mesajı ---
UYARI_HTML = '<p style="font-size: 0.7em; color: #888; margin-top: 10px; border-top: 1px solid #eee; padding-top: 5px;">⚠️ <strong>ÖNEMLİ:</strong> Bu yanıtlar hasta yorumlarına dayanır, tıbbi tavsiye değildir. Doktorunuza danışın.</p>'

# --- JSON Formatlama (Detaylı Yorumlar Desteği) ---
def format_rag_response_for_chat(response_string):
    try:
        json_match = re.search(r"(\{.*\})", response_string, re.DOTALL)
        if not json_match:
            return f"<div class='drug-container'><div class='info-card'><p>{response_string}</p></div></div>" + UYARI_HTML

        data = json.loads(json_match.group(0))
        ilac_info = data.get("Ilac_Hakkinda_Bilgi", "Bilgi bulunamadı.")
        genel_yorum = data.get("Genel_Yorumlar", "Yorum bulunamadı.")
        ortalama_puan = data.get("Ortalama_Puan", "Veri yok")
        detaylar = data.get("Detayli_Referanslar", [])

        score_card_html = f"""
        <div class="score-card">
            <h3>⭐ Ortalama Puan</h3>
            <p class="puan">{ortalama_puan}</p>
        </div>
        """

        html = f"""
        <div class="drug-container">
            <div class="info-card"><h3>💊 İlaç Hakkında Bilgi</h3><p>{ilac_info}</p></div>
            <div class="general-card"><h3>💬 Genel Yorumlar</h3><p>{genel_yorum}</p></div>
            <div class="review-card"><h3>📋 Detaylı Yorumlar</h3>
        """

        if isinstance(detaylar, list):
            if not detaylar:
                html += '<p class="yorum-yok">YETERLİ BİLGİ BULUNAMADI</p>'
            else:
                for ref in detaylar:
                    ilac = "Bilinmiyor"
                    puan = "-"
                    yorum = ""

                    # Dict formatındaki yorum
                    if isinstance(ref, dict):
                        ilac = ref.get("Ilac_Adi", "Bilinmiyor")
                        puan = ref.get("Puan", "-")
                        yorum = ref.get("Yorum", "")
                    # String formatındaki yorum
                    elif isinstance(ref, str):
                        pattern = re.compile(r"İlaç Adı:\s*(.*?)\.\s*Puan:\s*(.*?)\.\s*Yorum:\s*(.*)", re.DOTALL)
                        match = pattern.search(ref)
                        if match:
                            ilac = match.group(1).strip()
                            puan = match.group(2).strip()
                            yorum = match.group(3).strip()
                        else:
                            yorum = ref
                    else:
                        continue

                    yorum_temiz = yorum.strip().strip('"').strip("'").strip("“").strip("”")

                    html += f'<div class="review-item">'
                    baslik_satiri = ""
                    if ilac != "Bilinmiyor":
                        baslik_satiri = f"<strong>{ilac}</strong> — "
                    baslik_satiri += f'<span class="puan-mini">{puan}</span>'
                    html += f"<p>{baslik_satiri}</p>"
                    html += f'<p class="yorum">"{yorum_temiz}"</p>'
                    html += f'</div>'
        else:
            html += '<p class="yorum-yok">YETERLİ BİLGİ BULUNAMADI</p>'

        html += "</div>"
        html += score_card_html
        html += "</div>"

        return textwrap.dedent(html) + UYARI_HTML

    except Exception as e:
        st.code(response_string)
        return f"<div class='error-box'>⚠️ JSON işlenirken hata oluştu: {e}</div>"

# --- Chat Arayüzü ---
def display_chat_interface():
    st.set_page_config(page_title="🧠 MediMind – AI Health Assistant", layout="wide")
    apply_theme_js(st.session_state.theme_mode)
    load_css()

    col1, col2 = st.columns([8, 1])
    with col1: 
        st.title("🧠 MediMind – AI Health Assistant")
        st.caption("⚠️ **ÖNEMLİ:** Bu yapay zekâ asistanı, hasta yorumlarına dayanır. **Tıbbi tavsiye değildir.** Doktorunuza dananın.")
        
    with col2:
        btn_text = "☀️ Light" if st.session_state.theme_mode == "dark" else "🌙 Dark"
        st.button(btn_text, on_click=toggle_theme)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Merhaba! Hangi ilaç hakkında bilgi almak istersiniz?", "type": "text"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            if message.get("type") == "rag_response":
                formatted_content = format_rag_response_for_chat(content)
                st.markdown(formatted_content, unsafe_allow_html=True)
            else:
                st.markdown(content, unsafe_allow_html=True)

    # KRİTİK DÜZELTME KULLANILDI: 'client is None' yerine 'global_collection is None' kontrolü yapılıyor
    input_disabled = not bool(GEMINI_API_KEY) or global_collection is None
    if prompt := st.chat_input("💬 İlaç adını veya durumunu yazın...", disabled=input_disabled):
        st.session_state.messages.append({"role": "user", "content": prompt, "type": "text"})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Bilgi aranıyor..."):
                try:
                    raw_response = rag_answer_gemini(prompt)
                    formatted_content = format_rag_response_for_chat(raw_response)
                    st.markdown(formatted_content, unsafe_allow_html=True)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": raw_response, "type": "rag_response"}
                    )
                except Exception as e:
                    st.error(f"Hata: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Beklenmedik bir hata oluştu: {e}", "type": "text"})

# --- Uygulama Başlangıcı ---
display_chat_interface()