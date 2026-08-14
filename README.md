# Local RAG System (CPU-Based) (TR)

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
```
Bileşenler:
Doküman İşleme (ingestion.py): documents/ klasöründeki dosya ve rehberleri okur.

Akıllı Parçalama (chunker.py): Kelime bütünlüğünü koruyarak metinleri örtüşmeli (overlap) parçalara böler.

Vektör Arama (vector_store.py): Metin parçalarını embedding modeline iletir, JSON olarak SQLite'a kaydeder. Arama anında NumPy ile bellek içi (RAM) Kosinüs Benzerliği hesabı yapar.

Prompt & Çıkarım (prompt_templates.py & model_client.py): Bulunan bağlamı (context) şablona yerleştirip CPU üzerinde çalışan LLM'e gönderir.

## 📁 Proje Yapısı
Aşağıda projenin dosya ve klasör hiyerarşisi yer almaktadır:

```text
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
```

## ⚡ Kurulum ve Çalıştırma
## Sanal Ortam Oluşturma ve Bağımlılıklar
Projenin kök dizininde bir terminal açın ve aşağıdaki komutları sırasıyla çalıştırın:

### Sanal ortam oluşturma
python -m venv venv

### Sanal ortamı aktifleştirme (Windows)
venv\Scripts\activate

### Sanal ortamı aktifleştirme (Linux / macOS)
source venv/bin/activate

### Bağımlılıkları yükleme
pip install -r requirements.txt

## Modelleri İndirme
Yerel model kataloğundan gerekli LLM ve Embedding modellerini indirin:

python src/download_model.py

## Çıkarım Testi (CPU)
Modelin RAM'e sorunsuz yüklendiğini ve CPU üzerinde çalıştığını doğrulamak için:

python test_inference.py

## Uygulamayı Çalıştırma
Tüm RAG boru hattını terminal üzerinden başlatmak veya Streamlit arayüzünü açmak için:

### Terminal boru hattını çalıştırmak için
python main.py

### Streamlit arayüzünü açmak için
streamlit run app.py

## ⚙️ Öne Çıkan Özellikler
### Çevrimdışı ve Güvenli: 
Verileriniz dışarıya aktarılmaz, tüm işlemler bilgisayarınızda gerçekleşir.

### CPU Optimizasyonu: 
Grafik kartı (GPU) bağımlılığı olmadan, RAM ve CPU çekirdekleri üzerinde stabil çalışacak şekilde tasarlanmıştır.

### Hafif Vektör Arama: 
Ekstra ağır vektör veritabanları (Chroma, Milvus vb.) yerine SQLite ve NumPy C-API gücü kullanılır.

# Local RAG System (CPU-Based) (EN)

Foundry Local SDK-powered, fully local and offline Retrieval-Augmented Generation (RAG) architecture optimized for CPU performance.

This project processes local documents (PDF, TXT), performs vector search using SQLite and NumPy matrix operations without requiring an external vector database server, and generates context-aware responses via the Qwen 2.5 local language model.

---

### 🏗️ System Architecture

The diagram below shows the data flow:

```text
[ documents/ ] ──► (ingestion.py) ──► (chunker.py)
                                           │
                                           ▼
[ Qwen LLM (CPU) ] ◄── (model_client.py) ◄── (vector_store.py + SQLite)
```
Components:
Document Processing (ingestion.py): Reads files and guides from the documents/ directory.

Smart Chunking (chunker.py): Splits text into overlapping chunks while preserving word integrity.

Vector Search (vector_store.py): Sends text chunks to the embedding model and stores them as JSON in SQLite. Computes in-memory Cosine Similarity via NumPy during retrieval.

Prompt & Inference (prompt_templates.py & model_client.py): Injects the retrieved context into prompt templates and feeds it to the local LLM running on CPU.

## 📁 Project Structure
The project's file and folder hierarchy is provided below:

```text
LOCAL_RAG_PROJECT/
│
├── config/
│   └── __init__.py           # Configuration package directory
├── documents/                # Raw input documents (PDF, TXT)
│
├── src/
│   ├── __init__.py
│   ├── chunker.py            # Word-boundary aware text chunker
│   ├── download_model.py     # Model downloader and catalog manager
│   ├── ingestion.py          # Document loader and cleaner
│   ├── model_client.py       # Local LLM inference client
│   ├── prompt_templates.py   # RAG prompt templates
│   └── vector_store.py       # SQLite & NumPy based vector database
│
├── app.py                    # Streamlit GUI entry point
├── main.py                   # End-to-end RAG pipeline runner
├── test_inference.py         # CPU inference and hardware benchmark script
├── requirements.txt          # Project dependencies
└── .gitignore
```
## ⚡ Installation & Execution
## Creating a Virtual Environment and Dependencies
Open a terminal in the root directory of the project and run the following commands sequentially:

### Creating a virtual environment
python -m venv venv

### Activating the virtual environment (Windows)
venv\Scripts\activate

### Activating the virtual environment (Linux / macOS)
source venv/bin/activate

### Installing dependencies
pip install -r requirements.txt

## Downloading Models
Download the required LLM and Embedding models from the local model catalog:

python src/download_model.py

## Inference Test (CPU)
To verify that the model loads smoothly into RAM and runs on the CPU:

python test_inference.py

## Running the Application
To launch the complete RAG pipeline via terminal or open the Streamlit interface:

### To run the terminal pipeline
python main.py

### To open the Streamlit interface
streamlit run app.py

## ⚙️ Key Features
### Offline and Secure:
Your data is not transmitted externally; all operations take place locally on your computer.

### CPU Optimization:
Designed to run stably on RAM and CPU cores without requiring a graphics card (GPU).

### Lightweight Vector Search:
Utilizes the power of SQLite and NumPy C-API instead of heavy external vector databases (such as Chroma, Milvus, etc.).
