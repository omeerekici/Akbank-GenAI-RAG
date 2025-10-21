import chromadb
import os

# rag_module'da tanımlanan path'i kullanın
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "drugs_review_collection"


print("--- ChromaDB Koleksiyon Kontrolü ---")

# Veritabanı klasörünün varlığını kontrol et
if not os.path.exists(CHROMA_PATH) or not os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
    print(f"❌ Veritabanı dosyaları ({CHROMA_PATH}) bulunamadı.")
    print("Veritabanı henüz oluşturulmamış veya oluşturuluyor (Streamlit Cloud).")
else:
    try:
        # ChromaDB istemcisini oluşturduğunuz yolla başlatın
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # Veritabanındaki tüm koleksiyonları listeleyin
        all_collections = client.list_collections()
        
        if all_collections:
            print(f"✅ Toplam {len(all_collections)} koleksiyon bulundu:")
            found_target = False
            for col in all_collections:
                count = col.count()
                if col.name == COLLECTION_NAME:
                    print(f"   - Adı: {col.name}, Kayıt Sayısı: {count} ✅ (Doğru Koleksiyon)")
                    found_target = True
                else:
                    print(f"   - Adı: {col.name}, Kayıt Sayısı: {count} (Diğer Koleksiyon)")
            
            if not found_target:
                print(f"❌ '{COLLECTION_NAME}' koleksiyonu bulunamadı, ancak klasör var.")

        else:
            print("❌ Veritabanı klasörü var ancak içinde hiç koleksiyon bulunamadı.")
            
    except Exception as e:
        print(f"❌ KRİTİK HATA OLUŞTU: {e}")
        print("ChromaDB versiyonu veya dosya erişimi sorunu olabilir. (Kontrol amaçlı dosya)")