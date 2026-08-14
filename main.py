from src.vector_store import LocalEmbeddingClient, VectorDatabase 
from src.model_client import LocalModelClient 
import os
from src.chunker import TextChunker
import numpy as np
from src.prompt_templates import build_rag_prompt

def get_top_chunks(query: str, top_k: int = 4, db_path: str = "knowledge_base.db") -> list:

    """Kullanıcı sorgusuna anlamsal olarak en yakın doküman parçalarını getirir.

    Sorguyu bir embedding vektörüne dönüştürür. SQLite veritabanından çekilen tüm
    parçaların vektörleri ile sorgu vektörü arasında kosinüs benzerliği hesaplar.
    Sonuçları benzerlik skoruna göre azalan sırada sıralar ve en alakalı `top_k`
    kadar parçayı döndürür.

    Args:
        query (str): Kullanıcının aratmak istediği metin veya soru.
        top_k (int, optional): Getirilecek en yüksek skorlu metin parçası sayısı.
            Defaults to 4.
        db_path (str, optional): Vektör verilerinin bulunduğu SQLite veritabanı yolu.
            Defaults to "knowledge_base.db".

    Returns:
        list: Her bir elemanı 'id', 'doc_name', 'content' ve 'score' anahtarlarını
            içeren sözlüklerden (dict) oluşan en alakalı parçaların listesi.
            Veritabanı boşsa veya kayıt bulunamazsa boş liste (`[]`) döndürür.
    """
        
    print(f"\n (retieval) sorgu işleniyor: '{query}'")
    embed_client = LocalEmbeddingClient(model_name="qwen3-embedding-0.6b") 
    db = VectorDatabase(db_path=db_path) 

    
    raw_query_embed = embed_client.get_embedding(query) 

    if hasattr(raw_query_embed, "data") and len(raw_query_embed.data) > 0:  
        query_vector = raw_query_embed.data[0].embedding
    elif isinstance(raw_query_embed, dict) and "embedding" in raw_query_embed:
        query_vector = raw_query_embed["embedding"]
    else:
        query_vector = raw_query_embed

    query_vec_np = np.array(query_vector, dtype=np.float32) 
    all_chunks = db.fetch_all_chunks() 

    if not all_chunks: 
        print(" Veritabanında hiçbir kayıt bulunamadı")
        return [] 

    scored_chunks = [] 

    for chunk in all_chunks:
        doc_vector_np = np.array(chunk["embedding"], dtype=np.float32) 
        similarity_score = LocalEmbeddingClient.cosine_similarity(query_vec_np, doc_vector_np)

        scored_chunks.append({ 
            "id": chunk["id"],
            "doc_name": chunk["doc_name"],
            "content": chunk["content"],
            "score": similarity_score
        })
 
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k] 

def answer_query(query: str) -> str:

    """Kullanıcı sorusuna vektör araması ve yerel LLM kullanarak yanıt üretir.

    İşlem Adımları:
    1. `get_top_chunks` fonksiyonunu çağırarak en alakalı 5 bağlam parçasını getirir.
    2. Bağlam bulunamazsa kullanıcıya bilgi mesajı döndürür.
    3. `build_rag_prompt` ile sistem talimatlarını ve parçaları birleştirir.
    4. Hazırlanan metni `LocalModelClient` (Qwen 2.5 7B) modeline ileterek
       üretilen yanıtı döndürür.

    Args:
        query (str): Yanıtlanması istenen kullanıcı sorusu.

    Returns:
        str: Dil modeli tarafından üretilen cevap metni veya bilgi bulunamadı mesajı.
    """

    retrieved_chunks = get_top_chunks(query=query, top_k=5) 
    
    if not retrieved_chunks:
        return "Sağlanan dokümanlarda soruyla ilgili bir bilgi bulunamadı." 

    formatted_messages = build_rag_prompt(query=query, retrieved_chunks=retrieved_chunks)
    system_text = formatted_messages[0]["content"] 
    user_text = formatted_messages[1]["content"]
    plain_prompt = f"SİSTEM TALİMATI:\n{system_text}\n\n{user_text}"

    print(" yanıt üretiliyor...")
    model_client = LocalModelClient(model_name="qwen2.5-7b")
    response_text = model_client.generate_response(prompt=plain_prompt)

    return response_text 


def main():

    """Sürekli kullanıcı girdisi alan ve cevapları ekrana basan CLI döngüsünü çalıştırır.

    Kullanıcı 'q', 'quit', 'exit', 'cikis' veya 'çıkış' yazana kadar ya da
    `KeyboardInterrupt` (Ctrl+C) tetiklenene kadar döngü devam eder.
    """

    print("=" * 50)
    print("  RAG sistemi hazır  ")
    print("=" * 50)
    print("Sistem çalışıyor. Çıkmak için 'q' yazabilirsiniz.\n")

    while True:
        try:

            user_query = input("\n Soru Sorun: ").strip() 

            if user_query.lower() in ["q", "quit", "exit", "cikis", "çıkış"]:
                print("\nSistemden çıkılıyor...")
                break

            if not user_query:
                continue

            print("\n" + "-" * 50)
            
            final_answer = answer_query(query=user_query) 

            print("\n Cevap:")
            print("-" * 50)
            print(final_answer)
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n\nÇıkış yapıldı.")
            break
        except Exception as e:
            print(f"\n İşlem Hatası: {str(e)}")

if __name__ == "__main__":
    main()
