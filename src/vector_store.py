from foundry_local_sdk import Configuration, FoundryLocalManager 
import numpy  as np 
import json 
import sqlite3 


class LocalEmbeddingClient:

    """Foundry Local SDK kullanarak metinleri vektör temsiline (embedding) dönüştüren
    ve vektörler arası benzerlik hesaplayan istemci sınıfı."""


    def __init__(self, model_name: str = "qwen3-embedding-0.6b"): 
        
        self.model_name = model_name 
        
        if not hasattr(FoundryLocalManager, '_instance') or FoundryLocalManager._instance is None:
           
            try:
                self.config = Configuration(app_name="local_rag_project")
                FoundryLocalManager.initialize(self.config)

            except Exception:
                pass
                
        self.manager = FoundryLocalManager.instance 
        
        print(f"{self.model_name} modeli katalogdan alınıyor...") 
        self.model = self.manager.catalog.get_model(self.model_name) 
        self.model.load() 
        self.embed_client = self.model.get_embedding_client() 

    def get_embedding(self, text: str) -> list:

        """Verilen metni modele ileterek sayısal vektör dizisine (float listesi) dönüştürür.

        Args:
            text (str): Vektörleştirilecek ham metin.

        Returns:
            list: Metnin anlamsal vektör karşılığı (float dizisi).
        """

        try:
            response = self.embed_client.generate_embedding(text) 

            if hasattr(response, 'embedding'): 
                return response.embedding

            return response
        
        except Exception as e:
            print(f"Embedding üretilirken hata oluştu: {str(e)}")
            return [] 

    def compute_batch_similarity(self, query_vec: list, matrix: np.ndarray) -> np.ndarray:
        """Sorgu vektörünü normalize eder ve RAM'de önceden L2-normalize edilmiş

        (N, D) doküman matrisi ile tek hamlede matris çarpımı (Dot Product) yapar.

        Kosinüs Benzerliği = Q_norm . D_norm
        """
        query_np = np.array(query_vec, dtype=np.float32)

        q_norm = np.linalg.norm(query_np)
        query_normalized = query_np / (q_norm if q_norm != 0 else 1e-10)

        return np.dot(matrix, query_normalized)



    @staticmethod
    def cosine_similarity(vec1, vec2): 

        """İki vektör arasındaki anlamsal benzerliği Kosinüs Benzerliği formülü ile hesaplar.

        NumPy kullanarak noktasal çarpım ve L2 normu hesaplar.

        Args:
            vec1 (list): Birinci vektör dizisi.
            vec2 (list): İkinci vektör dizisi.

        Returns:
            float: 0.0 ile 1.0 arasında benzerlik skoru.
        """

        dot_product = np.dot(vec1, vec2) 
        norm_vec1 = np.linalg.norm(vec1) 
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
            
        return dot_product / (norm_vec1 * norm_vec2) 
    
class VectorDatabase:

    """SQLite altyapısını kullanarak doküman parçalarını (chunk) ve bunlara ait

    embedding vektörlerini saklayan ve sorgulayan vektör veritabanı sınıfı.
    """

    def __init__(self, db_path="knowledge_base.db"): 

        """Veri tabanı yolunu belirler ve gerekli tablo yapısını başlatır.

        Args:
            db_path (str): SQLite veri tabanı dosyasının yolu.
        """

        self.db_path = db_path
        self.init_db() 

    def init_db(self):

        """Veri tabanında 'document_chunks' tablosu yoksa oluşturur.

        Tablo yapısı: id, doc_name, content, embedding (JSON string).
        """
        
        conn = sqlite3.connect(self.db_path) 
        cursor = conn.cursor() 

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks ( 
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_name TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
        """)

        conn.commit() 
        conn.close() 

    def insert_chunk( self, doc_name: str, content: str, embedding_vector: list ) -> int:

        """Yeni bir doküman parçasını ve vektörünü serileştirerek veri tabanına kaydeder.

        Args:
            doc_name (str): Kaynak dokümanın adı.
            content (str): Parçalanmış metin içeriği (chunk).
            embedding_vector (list): Metne ait sayısal vektör listesi.

        Returns:
            int: Veri tabanına eklenen kaydın birincil anahtarı (ID).
        """

        conn = sqlite3.connect(self.db_path) 
        cursor = conn.cursor() 
        embedding_json = json.dumps(embedding_vector)

        cursor.execute(
            """
            INSERT INTO document_chunks (doc_name, content, embedding)
            VALUES (?, ?, ?) 
        """,
            (doc_name, content, embedding_json),
        )

        conn.commit() 
        inserted_id = cursor.lastrowid 
        conn.close()
        return inserted_id

    def fetch_all_chunks(self) -> list:

        """Veri tabanındaki tüm doküman parçalarını getirir ve JSON formatındaki

        embedding ifadelerini tekrar Python listesine dönüştürür.

        Returns:
            list[dict]: Her elemanı 'id', 'doc_name', 'content' ve 'embedding'
            anahtarlarını içeren sözlük listesi.
        """
        
        conn = sqlite3.connect(self.db_path) 
        cursor = conn.cursor() 
        
        cursor.execute( 
            "SELECT id, doc_name, content, embedding FROM document_chunks"
        )

        rows = cursor.fetchall() 
    
        processed_data = []
        for row in rows: 
            chunk_id, doc_name, content, embedding_str = row
            processed_data.append({
                "id": chunk_id,
                "doc_name": doc_name,
                "content": content,
                "embedding": json.loads(embedding_str), 
            })

        conn.close() 
        return processed_data 

