"""
test_inference.py

Bu betik, Foundry Local SDK kullanarak yerel LLM (Qwen 2.5-7B) modelinin 
başarıyla başlatıldığını, bellek yüklemesinin yapıldığını,
çevrimdışı çıkarım yeteneğini doğrulamak ve çıkarım esnasında donanımın hareketlerini gözlemlemek
amacıyla hazırlanmış test betiğidir. İşlem sonunda bellek otomatik olarak temizlenir.
"""

from foundry_local_sdk import Configuration, FoundryLocalManager

print("Foundry Local SDK Başlatılıyor...") 

try: 

    config = Configuration(app_name="local_rag_project")
    FoundryLocalManager.initialize(config) 
    manager = FoundryLocalManager.instance 

    print("Yerel LLM Modeli Katalogtan Alınıyor...")

    model = manager.catalog.get_model("qwen2.5-7b") 
    
    print("Model belleğe yükleniyor...")

    model.load() 
    
    print("\n--- Model Başarıyla Yüklendi! ---")
    print("Şimdi yerel modelimize bir soru soruyoruz. Görev yöneticisinden gözlem yapabilirsiniz.")
    print("-" * 50)
    
    chat_client = model.get_chat_client()
    
    prompt = "Bilgisayar mühendisliği öğrencileri için yapay zekanın önemi nedir? Kısa ve öz olarak açıkla."
    
    response = chat_client.complete_chat([ 
        {"role": "user", "content": prompt}
    ])

    print("\nYerel Yapay Zeka Cevabı:")
    print(response.choices[0].message.content) 
       
    model.unload()
    
    print("\nModel bellekten başarıyla kaldırıldı.")

except Exception as e: 
    print(f"\nBir hata oluştu: {e}")

input("\nTest bitti. Kapatmak için Enter'a basınız...") 
