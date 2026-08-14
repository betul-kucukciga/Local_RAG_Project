Markdown
# Local RAG System (CPU-Based)

Foundry Local SDK altyapısı kullanılarak tamamen yerel (local) ve çevrimdışı (offline) çalışan, CPU optimizasyonlu bir Retrieval-Augmented Generation (RAG) mimarisi.

Bu proje; harici bir vektör veritabanı sunucusuna ihtiyaç duymadan, yerel dokümanları (PDF, TXT) işler, SQLite ve NumPy matris operasyonları ile vektör araması yapar ve Qwen 2.5 yerel dil modeli üzerinden bağlama uygun yanıtlar üretir.

---

## 🏗️ Sistem Mimarisi

Aşağıdaki şema, veri akışını göstermektedir:

```text
[ documents/ ] ──► (ingestion.py) ──► (chunker.py)
                                           │
                                           ▼
[ Qwen LLM (CPU) ] ◄── (model_client.py) ◄── (vector_store.py + SQLite)
Bileşenler:
Doküman İşleme (ingestion.py): documents/ klasöründeki dosya ve rehberleri okur.

Akıllı Parçalama (chunker.py): Kelime bütünlüğünü koruyarak metinleri örtüşmeli (overlap) parçalara böler.

Vektör Arama (vector_store.py): Metin parçalarını embedding modeline iletir, JSON olarak SQLite'a kaydeder. Arama anında NumPy ile bellek içi (RAM) Kosinüs Benzerliği hesabı yapar.

Prompt & Çıkarım (prompt_templates.py & model_client.py): Bulunan bağlamı (context) şablona yerleştirip CPU üzerinde çalışan LLM'e gönderir.

📁 Proje Yapısı
Aşağıda projenin dosya ve klasör hiyerarşisi yer almaktadır:

Plaintext
LOCAL_RAG_PROJECT/
│
├── config/
│   └── __init__.py           # Yapılandırma paket dizini
├── documents/                # İşlenecek ham veriler (PDF, TXT rehberleri)
│   ├── internet ve ağ sorunu...
│   └── Saglik-Bakanligi-İlk-Y...
│
├── src/
│   ├── __init__.py
│   ├── chunker.py            # Kelime korumalı metin parçalama
│   ├── download_model.py     # Model indirici ve katalog yöneticisi
│   ├── ingestion.py          # Doküman okuma ve temizleme
│   ├── model_client.py       # Local LLM çıkarım istemcisi
│   ├── prompt_templates.py   # RAG prompt şablonları
│   └── vector_store.py       # SQLite & NumPy tabanlı vektör veritabanı
│
├── app.py                    # Arayüz / Uygulama giriş noktası (Streamlit)
├── main.py                   # RAG uçtan uca çalıştırma boru hattı (Pipeline)
├── test_inference.py         # CPU çıkarım ve donanım test betiği
├── requirements.txt          # Proje bağımlılıkları
└── .gitignore
⚡ Kurulum ve Çalıştırma
1. Sanal Ortam Oluşturma ve Bağımlılıklar
Projenin kök dizininde bir terminal açın ve aşağıdaki komutları sırasıyla çalıştırın:

Bash
# Sanal ortam oluşturma
python -m venv venv

# Sanal ortamı aktifleştirme (Windows)
venv\Scripts\activate

# Sanal ortamı aktifleştirme (Linux / macOS)
source venv/bin/activate

# Bağımlılıkları yükleme
pip install -r requirements.txt
2. Modelleri İndirme
Yerel model kataloğundan gerekli LLM ve Embedding modellerini indirin:

Bash
python src/download_model.py
3. Çıkarım Testi (CPU)
Modelin RAM'e sorunsuz yüklendiğini ve CPU üzerinde çalıştığını doğrulamak için:

Bash
python test_inference.py
4. Uygulamayı Çalıştırma
Tüm RAG boru hattını terminal üzerinden başlatmak veya Streamlit arayüzünü açmak için:

Bash
# Terminal boru hattını çalıştırmak için
python main.py

# Streamlit arayüzünü açmak için
streamlit run app.py
⚙️ Öne Çıkan Özellikler
🔒 Çevrimdışı ve Güvenli: Verileriniz dışarıya aktarılmaz, tüm işlemler bilgisayarınızda gerçekleşir.

💻 CPU Optimizasyonu: Grafik kartı (GPU) bağımlılığı olmadan, RAM ve CPU çekirdekleri üzerinde stabil çalışacak şekilde tasarlanmıştır.

🚀 Hafif Vektör Arama: Ekstra ağır vektör veritabanları (Chroma, Milvus vb.) yerine SQLite ve NumPy C-API gücü kullanılır.