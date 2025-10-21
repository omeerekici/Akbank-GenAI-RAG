import chromadb

import os



# ChromaDB istemcisini oluşturduğunuz yolla başlatın

client = chromadb.PersistentClient(path="./chroma_db")



print("--- Bulunan Koleksiyonlar ---")

try:

    # Veritabanındaki tüm koleksiyonları listeleyin

    all_collections = client.list_collections()

   

    if all_collections:

        print(f"Toplam {len(all_collections)} koleksiyon bulundu:")

        for col in all_collections:

            # Koleksiyonun adını, kimliğini ve sayısını yazdırın

            count = col.count()

            print(f"- Adı: {col.name}, ID: {col.id}, Kayıt Sayısı: {count}")

    else:

        print("❌ Veritabanında (./chroma_db) hiç koleksiyon bulunamadı.")

       

except Exception as e:

    print(f"❌ HATA OLUŞTU: {e}")

    print("ChromaDB versiyonu veya dosya erişimi sorunu olabilir.")