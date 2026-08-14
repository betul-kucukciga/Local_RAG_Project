"""
src/download_model.py
---------------------
Foundry Local SDK üzerinden model indirme ve katalog sorgulama modülü.
"""

from foundry_local_sdk import Configuration, FoundryLocalManager

project_name = "local_rag_project"


def _get_manager() -> FoundryLocalManager:

    """SDK yapılandırmasını başlatır ve aktif yönetici örneğini döndürür."""

    config = Configuration(app_name = project_name)
    FoundryLocalManager.initialize(config)
    return FoundryLocalManager.instance


def download_llm(manager: FoundryLocalManager, model_name: str = "qwen2.5-7b-instruct-generic-gpu:4") -> bool:

    """Ana yapay zeka dil modelini katalogdan bulur ve indirir."""

    try:
        model_meta = manager.catalog.get_model(model_name)

        if model_meta is not None:

            print(f"'{model_name}' indiriliyor...")
            model_meta.download()
            print(f"--- '{model_name}' başarıyla indirildi. ---")
            return True
        
        else:

            print(f"'{model_name}' katalogda bulunamadı.")
            return False
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return False


def download_embedding_model(manager: FoundryLocalManager, model_name: str = "qwen3-embedding-0.6b") -> bool:

    """Metinleri vektöre dönüştüren embedding modelini indirir."""

    try:

        model_meta = manager.catalog.get_model(model_name)

        if model_meta is not None:

            print(f"'{model_name}' indiriliyor...")
            model_meta.download()
            print(f"--- '{model_name}' başarıyla indirildi. ---")
            return True

        else:

            print(f"'{model_name}' katalogda bulunamadı.")
            return False

    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return False


def list_models_in_catalog(manager: FoundryLocalManager) -> list:

    """Katalogdaki tüm modelleri ekrana basar."""

    try:

        models = manager.catalog.list_models()

        if not models:

            print("Katalogda model bulunamadı.")
            return []

        for model in models:

            clean_name = (
                getattr(model, 'id', None) or 
                getattr(model, 'name', None) or 
                getattr(model, 'model_id', None) or 
                getattr(model, 'alias', None)
            )

            if not clean_name and isinstance(model, dict):

                clean_name = model.get('id') or model.get('name') or model.get('alias') or model.get('model_id')

            if not clean_name:

                clean_name = str(model).strip()

            print(f"- {clean_name}")

        return models

    except Exception as e:

        print(f"Hata oluştu: {str(e)}")
        return []

    
if __name__ == "__main__":
   
    manager = _get_manager()
    
    # ---Katalogtaki Modelleri Listelemek için # kaldırın---
    #list_models_in_catalog(manager)
    
    # ---LLM Modelini İndirmek için # kaldırın---
    # download_llm(manager, "qwen2.5-7b")
    
    # ---Embedding Modelini İndirmek için # kaldırın---
    # download_embedding_model(manager, "qwen3-embedding-0.6b") 