import os 
from src.chunker import TextChunker 
from src.vector_store import LocalEmbeddingClient, VectorDatabase 
import fitz 

def extract_text_from_pdf(pdf_path: str) -> str: 

    """PDF belgesini okur ve sayfa etiketleriyle birlikte tek bir string olarak döndürür.

    PyMuPDF (fitz) kütüphanesini kullanarak belgedeki tüm sayfaları sırayla tarar,
    boş olmayan sayfaların başına sayfa numarası etiketini ekler ve sayfaları
    çift satır atlamayla birleştirir.

    Args:
        pdf_path (str): Okunacak PDF dosyasının tam veya bağıl yolu.

    Returns:
        str: Sayfa bilgileriyle etiketlenmiş birleştirilmiş tüm PDF metni.
    """

    doc = fitz.open(pdf_path) 
    full_text = [] 
    
    for page_num in range(len(doc)):
        page = doc[page_num] 
      
        text = page.get_text("text") 
        if text.strip():
      
            full_text.append(f"--- Sayfa numarası: {page_num + 1} ---\n{text}") 
            
    doc.close()
    return "\n\n".join(full_text) 

def run_ingestion_pipeline(documents_dir: str = "documents", db_path: str = "knowledge_base.db"):

    """Dosya okuma, parçalama, embedding üretme ve DB kayıt adımlarını uçtan uca yürütür.

    Belirtilen klasördeki desteklenen (.pdf, .txt) dosyaları tarar. Her dosya için:
    1. Ham metni çıkarır.
    2. TextChunker ile küçük parçalara böler.
    3. LocalEmbeddingClient ile her parçanın vektör temsilini alır.
    4. Metni, kaynak dosya adını ve vektörü VectorDatabase'e kaydeder.

    Args:
        documents_dir (str, optional): İşlenecek belgelerin bulunduğu klasör dizini.
            Defaults to "documents".
        db_path (str, optional): Vektörlerin ve metinlerin kaydedileceği
            SQLite veritabanı dosya yolu. Defaults to "knowledge_base.db".

    Returns:
        None
    """

    if not os.path.exists(documents_dir): 
        os.makedirs(documents_dir)
        print(f"'{documents_dir}' klasörü oluşturuldu. Klasörün içine .pdf, .txt uzantılı dosyaları ekleyip tekrar çalıştırın.")
        return

    
    supported_extensions = (".pdf", ".txt") 
    files = [f for f in os.listdir(documents_dir) if f.lower().endswith(supported_extensions)]
    
    if not files: 
        print(f"'{documents_dir}' klasöründe işlenecek desteklenen (.pdf, .txt) dosya bulunamadı.")
        return

   
    embed_client = LocalEmbeddingClient(model_name="qwen3-embedding-0.6b") 
    db = VectorDatabase(db_path=db_path) 
    chunker = TextChunker(chunk_size=400, overlap=30) 

    total_chunks_processed = 0 

    for file_name in files: 
        file_path = os.path.join(documents_dir, file_name) 
        print(f"\n İşlenen dosya: {file_name}")

        
        raw_text = "" 
        if file_name.lower().endswith(".pdf"): 
            
            raw_text = extract_text_from_pdf(file_path)
        elif file_name.lower().endswith(".txt"):
            
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

       
        chunks = chunker.chunk_text(raw_text)  
        print(f"Toplam {len(chunks)} parçaya bölündü.")

        for idx, chunk_text in enumerate(chunks): 
            
            raw_response = embed_client.get_embedding(chunk_text)

            
            if hasattr(raw_response, "data") and len(raw_response.data) > 0:
                embedding_vector = raw_response.data[0].embedding
            elif isinstance(raw_response, dict) and "embedding" in raw_response:
                embedding_vector = raw_response["embedding"]
            else:
                embedding_vector = raw_response

            
            inserted_id = db.insert_chunk(
                doc_name=file_name,
                content=chunk_text,
                embedding_vector=embedding_vector
            )
            total_chunks_processed += 1
            print(f"  [parça: {idx + 1}/{len(chunks)}] -> SQLite DB Kayıt ID: {inserted_id}")

    print("\n--- Veritabanına metinler başarıyla aktarıldı ---")
    print(f"Toplam {len(files)} dosya işlendi, {total_chunks_processed} parça veritabanına kaydedildi.")

if __name__ == "__main__":
    run_ingestion_pipeline()
