from foundry_local_sdk import Configuration, FoundryLocalManager

class LocalModelClient:
    """Foundry Local SDK kullanarak yerel yapay zeka dil modellerini (LLM) yöneten istemci sınıfı.
    
    Attributes:
        model_name (str): Belleğe yüklenecek olan yerel LLM modelinin adı.
        manager (FoundryLocalManager): SDK'nın aktif sistem yöneticisi nesnesi.
        model: Katalogdan çağrılan ve belleğe yüklenen model nesnesi.
        chat_client: Model ile sohbet etmek için kullanılan SDK sohbet istemcisi.
    """

    def __init__(self, model_name: str = "qwen2.5-7b"):
        """Sistemi ilklendirir, belirtilen modeli kataloğundan çekip RAM'e yükler.

        Args:
            model_name (str, optional): Yüklenecek modelin katalog adı.
                Varsayılan değeri "qwen2.5-7b-generic-cpu-4"'dir.
        """
        self.model_name = model_name

        # DÜZELTME: Doğrudan SDK'nın instance durumunu kontrol ediyoruz.
        if FoundryLocalManager.instance is None:
            config = Configuration(app_name="local_rag_project")
            try:
                FoundryLocalManager.initialize(config)
            except Exception as e:
                raise RuntimeError(f"FoundryLocalManager ilklendirilemedi: {str(e)}")

        self.manager = FoundryLocalManager.instance
        
        # Manager'ın başarılı bir şekilde elde edildiğinden emin oluyoruz
        if self.manager is None:
            raise RuntimeError("FoundryLocalManager örneği (instance) alınamadı.")

        self.model = self.manager.catalog.get_model(self.model_name)
        self.model.load()
        self.chat_client = self.model.get_chat_client()

    def generate_response(self, prompt: str) -> str:
        """Kullanıcıdan gelen metinsel istemi (prompt) modele iletir ve yanıt üretir.

        Args:
            prompt (str): Modele sorulacak soru veya verilecek ana talimat.

        Returns:
            str: Modelin ürettiği düz metin yanıtı.
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.chat_client.complete_chat(messages)
            return response.choices[0].message.content

        except Exception as e:
            return f"Model çıkarımı esnasında hata oluştu: {str(e)}"