import google.generativeai as genai



from sentence_transformers import SentenceTransformer



import chromadb



import os



from dotenv import load_dotenv







# --- API Anahtarı ve Konfigürasyon ---



load_dotenv()



GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")







if GEMINI_API_KEY:



    genai.configure(api_key=GEMINI_API_KEY)



else:



    print("❌ UYARI: GEMINI_API_KEY çevresel değişkende bulunamadı.")











# --- Sorgu Genişletme Fonksiyonu ---







def expand_query(query):



    """Bazı yaygın ticari adları etken madde ile genişletir."""



    query_lower = query.lower()



   



    # Veritabanınızın bulma olasılığını artırmak için eşanlamlılar



    if "accutane" in query_lower:



        query += " Roaccutane Isotretinoin Zoretanin"



    elif "zoloft" in query_lower:



        query += " Sertralin"



    elif "xanax" in query_lower:



        query += " Alprazolam"



    elif "valium" in query_lower:



        query += " Diazepam"



   



    return query











# --- Chroma ve embedding modeli yeniden yükle (Global değişkenler) ---







# Değişkenleri HATA DURUMUNDA bile erişilebilir kılmak için başta tanımlıyoruz



client = None



collection = None







try:



    model = SentenceTransformer('all-MiniLM-L6-v2')



    client = chromadb.PersistentClient(path="./chroma_db")



    collection = client.get_collection("drugs_review_collection")



    print("✅ RAG Modülü: ChromaDB ve Embeddings başarıyla yüklendi.")



except Exception as e:



    # Hata oluşsa bile client ve collection değişkenleri None olarak tanımlı kalır (Yukarıda tanımlandığı gibi)



    print(f"❌ RAG Modülü HATA: ChromaDB yüklenemedi. Detay: {e}")



    pass # Hata fırlatmak yerine None olarak kalmasını sağlıyoruz











def retrieve(query, k=10):



    """Sorguya en yakın dokümanları ChromaDB'den çeker."""



    if collection is None:



        return {'documents': [["[HATA] ChromaDB koleksiyonu yüklenemedi."]]}



       



    try:



        q_emb = model.encode([query])[0]



        results = collection.query(



            query_embeddings=[q_emb.tolist()],



            n_results=k,



            include=['documents', 'metadatas', 'distances']



        )



        return results



    except Exception as e:



        return {'documents': [[f"[HATA] Embeddings veya sorgulama başarısız oldu: {e}"]]}











def rag_answer_gemini(query, k=10, model_name="gemini-2.5-flash"):



    """Çekilen verileri kullanarak veya Google Search ile yanıt alır."""



   



    # --- Genel Sorgu Kontrolü (Google Search Kullanımı) ---



    if "benzer ilaçlar" in query.lower() or "ne işe yarar" in query.lower() or "sınıflandırması" in query.lower():



       



        prompt_genel = f"Soru: {query}. Bu soruyu cevaplamak için Google Arama aracını kullan. Cevabını sadece düz metin olarak ver. Tıbbi tavsiye verme."



       



        try:



            model = genai.GenerativeModel(model_name)



            # Google Arama aracını ekle



            response = model.generate_content(



                prompt_genel,



                tools=[{"googleSearch": {}}]



            )



            return response.text



        except Exception as e:



            return f"Genel sorgu başarısız: {e}"



           



    # --- RAG Sorgusu (Hasta Yorumları) ---



    else:



        # Sorguyu genişletme



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