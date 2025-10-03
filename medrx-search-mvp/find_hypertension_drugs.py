import sys
import pandas as pd
sys.path.append('src')

print("=== ПОШУК ПРЕПАРАТІВ ВІД ГІПЕРТЕНЗІЇ В ПОВНОМУ ДАТАСЕТІ ===")

# Завантажуємо повний датасет
df = pd.read_parquet("/home/anatolii-shara/Documents/scraping_compendium/compendium_all.parquet")
print(f"Повний датасет: {len(df)} препаратів")

# Шукаємо препарати з гіпертензією
hypertension_terms = ['гіпертензі', 'гіпертонічн', 'високий тиск', 'підвищений тиск', 'артеріальний тиск']

found_drugs = []
for idx, row in df.iterrows():
    indications = str(row.get('Показання', '')).lower()
    drug_name = str(row.get('Назва препарату', ''))
    
    for term in hypertension_terms:
        if term in indications:
            found_drugs.append({
                'index': idx,
                'name': drug_name,
                'indications': row.get('Показання', '')[:200] + '...'
            })
            break

print(f"\n🎯 ЗНАЙДЕНО {len(found_drugs)} препаратів від гіпертензії:")

# Показуємо перші 10 знайдених
for i, drug in enumerate(found_drugs[:10]):
    print(f"{i+1}. {drug['name']} (індекс {drug['index']})")
    print(f"   Показання: {drug['indications']}")
    print()

if found_drugs:
    # Створюємо тестовий датасет з препаратами від гіпертензії
    hypertension_indices = [drug['index'] for drug in found_drugs[:20]]
    test_df = df.iloc[hypertension_indices].copy()
    
    print(f"✅ Створюємо тестовий датасет з {len(test_df)} препаратів від гіпертензії")
    
    # Тестуємо enhanced assistant на релевантних даних
    from search.enhanced_medical_assistant import EnhancedMedicalAssistant, SearchConfig
    
    print("\n[ТЕСТ] Enhanced Medical Assistant на релевантних даних:")
    assistant = EnhancedMedicalAssistant()
    assistant.build_from_dataframe(test_df, encoder_model=None, medical_chunking=True, max_chunk_tokens=128)
    
    # Пошук
    config = SearchConfig(show=5)
    results = assistant.search("артеріальна гіпертензія", config)
    
    print(f"\nРезультати пошуку:")
    for i, drug in enumerate(results):
        print(f"{i+1}. {drug['drug_name']}")
        print(f"   Score: {drug['best_score']:.3f}")
        print(f"   Passages: {len(drug['passages'])}")

else:
    print("❌ Препарати від гіпертензії не знайдено в датасеті!")

