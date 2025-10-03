import sys
import pandas as pd
sys.path.append('src')

from search.enhanced_medical_assistant import EnhancedMedicalAssistant, SearchConfig

print("=== ТЕСТ НА РЕАЛЬНИХ МЕДИЧНИХ ДАНИХ ===")

try:
    # Завантажуємо невелику частину даних
    df = pd.read_parquet("/home/anatolii-shara/Documents/scraping_compendium/compendium_all.parquet")
    print(f"Завантажено {len(df)} препаратів")
    
    # Берем перші 50 для швидкого тестування
    test_df = df.head(50).copy()
    print(f"Тестуємо на {len(test_df)} препаратах")
    
    # Створюємо Enhanced assistant
    assistant = EnhancedMedicalAssistant()
    
    # Build index з medical chunking
    print("\n[1] Будуємо індекс з medical chunking...")
    assistant.build_from_dataframe(
        test_df, 
        encoder_model=None,  # поки без FAISS для швидкості
        medical_chunking=True,
        max_chunk_tokens=256
    )
    
    print(f"Створено {len(assistant.passages)} passages")
    print(f"Перші 3 passages:")
    for i in range(min(3, len(assistant.passages))):
        p = assistant.passages[i]
        print(f"  {i}: {p[:100]}... ({len(p)} символів)")
    
    # Тестуємо пошук
    test_queries = [
        "артеріальна гіпертензія",
        "серцева недостатність", 
        "головний біль"
    ]
    
    config = SearchConfig(show=5)
    
    for query in test_queries:
        print(f"\n[ПОШУК] '{query}':")
        results = assistant.search(query, config)
        print(f"Знайдено {len(results)} препаратів")
        
        for i, drug in enumerate(results[:3]):
            print(f"  {i+1}. {drug['drug_name']}")
            print(f"     Score: {drug['best_score']:.3f}")
            print(f"     Passages: {len(drug['passages'])}")
            print(f"     Risk level: {drug.get('risk_level', 'N/A')}")
    
    print("\n✅ ТЕСТ НА РЕАЛЬНИХ ДАНИХ ПРОЙШОВ УСПІШНО!")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

