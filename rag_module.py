import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
import os
from dotenv import load_dotenv
import streamlit as st 
import pandas as pd 
import numpy as np 

# --- Sabitler ve Konfigürasyon ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "drugs_review_collection"
# Ham veri dosyalarınızın bulunduğu klasör
DATA_SOURCE_PATH = "./data_sources" 


if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("❌ UYARI: GEMINI_API_KEY çevresel değişkende bulunamadı. Lütfen Streamlit secrets'a ekleyin.")


# --- 1. Veritabanını Koşullu Yükleme/Oluşturma Fonksiyonu ---

@st.cache_resource
def get_chroma_db():
    print("--- ChromaDB Başlatılıyor/Kontrol Ediliyor ---")
    
    # Gömme (Embedding) Modelini Her Zaman Yarat
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"❌ Embeddings Modeli Yüklenemedi: {e}")
        return None, None 

    # Kontrol: Klasör ve ChromaDB'nin ana dosyası (chroma.sqlite3) var mı?
    db_exists = os.path.exists(CHROMA_PATH) and os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3"))

    if db_exists:
        # --- DURUM 1: Var Olanı Yükle ---
        try:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            collection = client.get_collection(COLLECTION_NAME)
            print("✅ ChromaDB ve Embeddings başarıyla yüklendi.")
            return collection, model
        except Exception as e:
            print(f"❌ RAG Modülü HATA: Yükleme başarısız. Hata: {e}")
            return None, None
            
    else:
        # --- DURUM 2: Yoksa Oluştur (Streamlit Cloud'da İlk Çalıştırma) ---
        print("Veritabanı bulunamadı. Ham veriden yeniden oluşturuluyor...")
        
        try:
            
            # *******************************************************************
            ### VERİ YÜKLEME VE İŞLEME MANTIĞI: RAM & ENCODING İÇİN DÜZELTİLDİ ###
            # *******************************************************************
            
            # 1. Ham Veriyi Yükleme ve Birleştirme
            train_file = os.path.join(DATA_SOURCE_PATH, 'drugsComTrain_raw.csv')
            test_file = os.path.join(DATA_SOURCE_PATH, 'drugsComTest_raw.csv')
            
            if not os.path.exists(train_file) or not os.path.exists(test_file):
                 raise FileNotFoundError(f"Ham veri dosyalarından biri veya ikisi bulunamadı. Lütfen {DATA_SOURCE_PATH} klasörünü kontrol edin.")

            # --- VERİ OKUMA VE BİRLEŞTİRME ---
            # KRİTİK DÜZELTME: Unicode hatası için 'latin-1' encoding kullanıldı.
            # RAM nedeniyle sadece TRAIN setini okuyoruz.
            df_train = pd.read_csv(train_file, encoding='latin-1') 
            df = df_train # Sadece train setini kullanıyoruz
            
            # 2. Veri Temizleme ve Hazırlık
            REQUIRED_COLUMNS = ['review', 'drugName', 'condition', 'rating']
            df.dropna(subset=REQUIRED_COLUMNS, inplace=True)
            
            # *******************************************************
            # KRİTİK KISITLAMA: RAM HATASINI ÇÖZMEK İÇİN 50K SINIRLANDIRILDI
            # *******************************************************
            LIMIT_ROW_COUNT = 50000 
            if len(df) > LIMIT_ROW_COUNT:
                df = df.head(LIMIT_ROW_COUNT) # İlk 50K satıra indir
            
            print(f"⚠️ Bellek koruması için sadece {len(df)} kayıt kullanılacaktır.")
            
            # Metinler (documents) ve Metadatalar (passages_df)
            texts = df['review'].tolist()
            passages_df = df[['drugName', 'condition', 'rating']].copy()
            
            # 3. Embeddings'i Yeniden Hesaplama
            print(f"▶️ Toplam {len(texts)} kayıt için Embeddings yeniden hesaplanıyor...")
            embeddings = model.encode(texts, show_progress_bar=False) # Logları sadeleştirdik
            print("✅ Embeddings hesaplaması tamamlandı.")
            
            # 4. ChromaDB'yi Başlatma ve Oluşturma
            client = chromadb.PersistentClient(path=CHROMA_PATH) 
            
            try:
                client.delete_collection(name=COLLECTION_NAME)
            except:
                pass
            
            collection = client.create_collection(name=COLLECTION_NAME, metadata={"source":"drugsCom"})

            # 5. Verileri Toplu Halde Ekleme
            ids = [f"passage_{i}" for i in range(len(texts))]
            metadatas = passages_df.to_dict(orient='records') 

            batch_size = 500
            for i in range(0, len(ids), batch_size):
                j = min(i + batch_size, len(ids))
                embeddings_batch = [e.tolist() for e in embeddings[i:j]] 
                collection.add(
                    ids=ids[i:j],
                    documents=texts[i:j],
                    metadatas=metadatas[i:j],
                    embeddings=embeddings_batch
                )

            print(f"✅ ChromaDB'ye veri eklendi. Toplam kayıt: {collection.count()}")
            
            return collection, model
            
        except Exception as e:
            # Hata detayını terminalde gösteriyoruz, böylece neyin hata verdiğini anlarız.
            print(f"❌ RAG Modülü KRİTİK HATA: Veritabanı oluşturulamadı. Detay: {e}")
            return None, None

# Fonksiyonu çalıştırarak global değişkenleri doldurun
global_collection, global_model = get_chroma_db()
print(f"Chroma durumu: {global_collection is not None}") 


# --- Sorgu Genişletme Fonksiyonu (Aynı kalır) ---
def expand_query(query):
    """Bazı yaygın ticari adları etken madde ile genişletir."""
    query_lower = query.lower()
    if "accutane" in query_lower:
        query += " Roaccutane Isotretinoin Zoretanin"
    elif "zoloft" in query_lower:
        query += " Sertralin"
    elif "xanax" in query_lower:
        query += " Alprazolam"
    elif "valium" in query_lower:
        query += " Diazepam"
    return query


# --- retrieve Fonksiyonu (Aynı kalır) ---
def retrieve(query, k=10):
    """Sorguya en yakın dokümanları ChromaDB'den çeker."""
    if global_collection is None:
        return {'documents': [["[HATA] ChromaDB koleksiyonu yüklenemedi. (Veritabanı Oluşturma Hatası)"]]}
        
    try:
        q_emb = global_model.encode([query])[0]
        results = global_collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        return results
    except Exception as e:
        return {'documents': [[f"[HATA] Embeddings veya sorgulama başarısız oldu: {e}"]]}

# --- rag_answer_gemini Fonksiyonu (Aynı kalır) ---
def rag_answer_gemini(query, k=10, model_name="gemini-2.5-flash"):
    """Çekilen verileri kullanarak veya Google Search ile yanıt alır."""
    
    # --- Genel Sorgu Kontrolü (Google Search Kullanımı) ---
    if "benzer ilaçlar" in query.lower() or "ne işe yarar" in query.lower() or "sınıflandırması" in query.lower():
        prompt_genel = f"Soru: {query}. Bu soruyu cevaplamak için Google Arama aracını kullan. Cevabını sadece düz metin olarak ver. Tıbbi tavsiye verme."
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt_genel,
                tools=[{"googleSearch": {}}]
            )
            return response.text
        except Exception as e:
            return f"Genel sorgu başarısız: {e}"
            
    # --- RAG Sorgusu (Hasta Yorumları) ---
    else:
        expanded_query = expand_query(query)
        res = retrieve(expanded_query, k=10)
        contexts = "\n\n---\n".join(res['documents'][0])
        
        if "[HATA]" in contexts:
            return contexts
            
        prompt_rag = f"""Kullanıcı bir ilaç hakkında bilgi istiyor: "{query}".
Aşağıda bu ilaçla ilgili hasta yorumları, kullanım durumları ve puanlar yer alıyor.

Aşağıdaki referans bilgilere dayanarak, sorulan konu hakkında bulunan ilaçları ve yorumları özetle.
Eğer referans bilgilerde, sorulan konu ("{query}") hakkında doğrudan ve yeterli bilgi bulunmuyorsa, bu durumu 'Ilac_Hakkinda_Bilgi' alanında belirtip diğer alanlara 'YETERLİ BİLGİ BULUNAMADI' yazın.

Cevabını aşağıdaki **KESİNLİKLE JSON formatında** oluştur.
Tıbbi tavsiye verme, sadece referans bilgilere dayanarak özet yap.

Gerekli JSON Anahtarları ve Açıklamaları:
1. 'Ilac_Hakkinda_Bilgi': İlacın ne için kullanıldığına dair kısa bir özet (Maksimum 2 cümle).
2. 'Genel_Yorumlar': Hastaların genel memnuniyetini ve olası yaygın yan etkileri özetleyen bir paragraf.
3. 'Ortalama_Puan': Bulunan yorumlardaki puanların ortalaması (örneğin: 7.5/10). Eğer ortalama hesaplanamıyorsa, 'Veri Yetersiz' yaz.
4. 'Detayli_Referanslar': En yakın 3 yorumu, ilacın adıyla ve puanıyla (Örn: "İlaç Adı: X. Puan: 8/10. Yorum:...") listeleyen bir dizi (list).

---
Referans Bilgiler:
{contexts}
---
Soru: {query}
CEVAP (Sadece JSON):"""

        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt_rag)
            return response.text
        except Exception as e:
            return f"{{'Hata': 'Modelden yanıt alınamadı.', 'Detay': '{e}'}}"