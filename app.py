"""
Yerel RAG (Retrieval-Augmented Generation) Yapay Zeka Asistanı Streamlit Arayüzü.

Bu modül, kullanıcı sorularını alır, RAM üzerinde önbelleklenmiş vektör indeksini
kullanarak en alakalı doküman parçalarını sorgular ve yerel dil
modeline bağlam sağlayarak yanıtlar üretir.
"""

import os
import time
import numpy as np
import streamlit as st

from src.model_client import LocalModelClient
from src.prompt_templates import build_rag_prompt
from src.vector_store import LocalEmbeddingClient, VectorDatabase

st.set_page_config(page_title="Yerel RAG Asistanı", page_icon="🔍")
st.title("Yerel RAG Yapay Zeka Asistanı")

with st.sidebar:
    if st.button("Uygulamayı Kapat"):
        st.write("Kapatılıyor...")
        os._exit(0)


@st.cache_resource
def load_embedding_client() -> LocalEmbeddingClient:
    """Yerel embedding model istemcisini oluşturur ve önbelleğe alır.

    Streamlit `@st.cache_resource` dekoratörü sayesinde model yalnızca uygulama
    ilk açıldığında bir kez yüklenir, sonraki isteklerde RAM'deki örnek yeniden kullanılır.

    Returns:
        LocalEmbeddingClient: Yüklenmiş metin gömme (embedding) istemcisi nesnesi.
    """
    return LocalEmbeddingClient(model_name="qwen3-embedding-0.6b")


@st.cache_resource
def load_model_client() -> LocalModelClient:
    """Yerel Büyük Dil Modeli (LLM) istemcisini oluşturur ve önbelleğe alır.

    Returns:
        LocalModelClient: Metin üretimi yapacak yerel dil modeli istemcisi nesnesi.
    """
    return LocalModelClient(model_name="qwen2.5-7b")


@st.cache_resource
def load_vector_db() -> VectorDatabase:
    """SQLite tabanlı vektör veritabanı bağlantı nesnesini oluşturur ve önbelleğe alır.

    Returns:
        VectorDatabase: Veritabanı sorgu ve veri çekme işlemlerini yürüten nesne.
    """
    return VectorDatabase(db_path="knowledge_base.db")

@st.cache_resource
def get_vector_index(_db: VectorDatabase):
    """Veritabanındaki tüm vektörleri RAM'e yükler, matrise dönüştürür ve normalize eder.

    Veritabanındaki tüm doküman parçalarını (chunks) çeker, ham embedding dizilerini
    (N, D) boyutunda 2D NumPy matrisine dönüştürür. Ardından her bir vektörün L2 normunu
    hesaplayarak uzunluklarını 1 birime eşitler. Sonuçlar RAM'de tutulur.

    Args:
        _db (VectorDatabase): SQLite vektör veritabanı bağlantı nesnesi. (Başındaki '_'
            simgesi Streamlit'in bu parametreyi hash'lemesini engeller).

    Returns:
        Tuple[Optional[np.ndarray], list]:
            - normalized_matrix (Optional[np.ndarray]): (N, D) boyutunda normalize edilmiş
              vektör matrisi. Veritabanı boşsa None döner.
            - all_chunks (list): Veritabanındaki doküman parçalarının metadata listesi.
    """
    all_chunks = _db.fetch_all_chunks()
    if not all_chunks:
        return None, []

   
    raw_matrix = np.array(
        [chunk["embedding"] for chunk in all_chunks], dtype=np.float32
    )

    norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10 
    normalized_matrix = raw_matrix / norms

    return normalized_matrix, all_chunks

embed_client = load_embedding_client()
model_client = load_model_client()
db = load_vector_db()

def answer_query(query: str):
    """Kullanıcı sorgusunu alır, vektör aramasını yürütür ve LLM ile yanıt üretir.

    Sorgunun vektör karşılığını alır, önbellekteki doküman matrisiyle toplu
    vektör çarpımı yaparak benzerlik skorlarını hesaplar. Eşik
    değerin üstündeki en alakalı parçaları (top-4) seçer ve prompt şablonuna
    yerleştirerek LLM'den yanıt üretir.

    Args:
        query (str): Kullanıcı tarafından arayüzden girilen metin sorgusu.

    Returns:
        Tuple[str, List[dict], float, float]:
            - response_text (str): LLM tarafından üretilen yanıt metni.
            - top_chunks (List[dict]): Yanıt oluştururken kullanılan kaynak parçalar.
            - search_duration (float): Vektör arama işleminin sürdüğü süre (saniye).
            - llm_duration (float): LLM modelinin yanıt üretme süresi (saniye).
    """
    start_search_time = time.perf_counter()
    similarity_threshold = 0.30

    matrix, chunks = get_vector_index(db)

    if matrix is None or len(chunks) == 0:
        search_duration = time.perf_counter() - start_search_time
        return (
            "Sağlanan dokümanlarda soruyla ilgili bir bilgi bulunamadı.",
            [],
            search_duration,
            0.0,
        )

    
    raw_query_embed = embed_client.get_embedding(query)
    query_vector = (
        raw_query_embed.data[0].embedding
        if hasattr(raw_query_embed, "data")
        else raw_query_embed
    )

    
    scores = embed_client.compute_batch_similarity(query_vector, matrix)

    scored_chunks = []
    for idx, score in enumerate(scores):
        if score >= similarity_threshold:
            chunk_copy = chunks[idx].copy()
            chunk_copy["score"] = float(score)
            scored_chunks.append(chunk_copy)

    search_duration = time.perf_counter() - start_search_time

    if not scored_chunks:
        return (
            "Bu bilgi sağlanan dokümanlarda bulunmamaktadır.",
            [],
            search_duration,
            0.0,
        )

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = scored_chunks[:4]


    formatted_messages = build_rag_prompt(
        query=query, retrieved_chunks=top_chunks
    )
    system_text = formatted_messages[0]["content"]
    user_text = formatted_messages[1]["content"]
    plain_prompt = f"SİSTEM TALİMATI:\n{system_text}\n\n{user_text}"

    start_llm_time = time.perf_counter()
    response_text = model_client.generate_response(prompt=plain_prompt)
    llm_duration = time.perf_counter() - start_llm_time

    print("\n--- Performans Analizi ---")
    print(f"Vektör Arama Süresi : {search_duration:.6f} saniye")
    print(f"LLM Yanıt Süresi     : {llm_duration:.4f} saniye")
    print(f"Toplam Süre          : {(search_duration + llm_duration):.4f} saniye\n")

    return response_text, top_chunks, search_duration, llm_duration


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Kullanılan Kaynaklar:"):
                for idx, src in enumerate(message["sources"], 1):
                    st.caption(
                        f"Kaynak {idx}: {src['doc_name']} (Benzerlik Skoru:"
                        f" {src.get('score', 0):.2f})"
                    )
                    st.text(src["content"])

user_query = st.chat_input("Bir soru sorun...")

if user_query:
    clean_query = user_query.strip()

    if not clean_query:
        st.warning(
            "Lütfen sadece boşluk bırakmak yerine geçerli bir soru yazın."
        )
    else:

        st.session_state.messages.append(
            {"role": "user", "content": clean_query}
        )
        with st.chat_message("user"):
            st.markdown(clean_query)


        with st.chat_message("assistant"):
            with st.spinner("Dokümanlar taranıyor ve yanıt üretiliyor..."):
                ai_response, sources, t_search, t_llm = answer_query(
                    clean_query
                )
                st.markdown(ai_response)

                total_t = t_search + t_llm
                st.caption(
                    f"Toplam Yanıt Süresi: **{total_t:.2f} s** (Arama:"
                    f" {t_search:.4f}s | LLM: {t_llm:.2f}s)"
                )

                if sources:
                    with st.expander("Kullanılan Kaynaklar:"):
                        for idx, src in enumerate(sources, 1):
                            st.caption(
                                f"Kaynak {idx}: {src['doc_name']} (Benzerlik"
                                f" Skoru: {src['score']:.2f})"
                            )
                            st.text(src["content"])


        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_response,
            "sources": sources,
        })
