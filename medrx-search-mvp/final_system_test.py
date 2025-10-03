import sys
import pandas as pd
sys.path.append('src')

from search.enhanced_medical_assistant import EnhancedMedicalAssistant, SearchConfig

print("=== ФІНАЛЬНЕ ТЕСТУВАННЯ ПОКРАЩЕНОЇ СИСТЕМИ ===")

# Завантажуємо повний датасет
df = pd.read_parquet("/home/anatolii-shara/Documents/scraping_compendium/compendium_all.parquet")

# Створюємо збалансовану вибірку з різними захворюваннями
balanced_terms = {
    'гіпертензі': 30,      # гіпертензія
    'серцев недостатн': 30, # серцева недостатність  
    'діабет': 25,          # цукровий діабет
    'головн біл': 20,      # головний біль
    'астм': 15,            # астма
}

selected_indices = []
for term, max_count in balanced_terms.items():
    count = 0
    for idx, row in df.iterrows():
        if count >= max_count:
            break
        indications = str(row.get('Показання', '')).lower()
        if term in indications and idx not in selected_indices:
            selected_indices.append(idx)
            count += 1

test_df = df.iloc[selected_indices].copy()
print(f"Створено збалансований тестовий датасет: {len(test_df)} препаратів")

# Тестуємо з оптимальними параметрами
print("\n[1] ENHANCED ASSISTANT з оптимізацією:")
assistant = EnhancedMedicalAssistant()
assistant.build_from_dataframe(
    test_df, 
    encoder_model=None,
    medical_chunking=True,
    max_chunk_tokens=100  # Зменшуємо для уникнення warning
)

# Тестові запити
test_queries = [
    "артеріальна гіпертензія",
    "серцева недостатність", 
    "цукровий діабет",
    "головний біль",
    "астма"
]

config = SearchConfig(show=3)

for query in test_queries:
    print(f"\n{'='*50}")
    print(f"🔍 ЗАПИТ: '{query}'")
    print(f"{'='*50}")
    
    results = assistant.search(query, config)
    
    if results:
        for i, drug in enumerate(results):
            print(f"{i+1}. {drug['drug_name']}")
            print(f"   📈 Score: {drug['best_score']:.3f}")
            print(f"   📋 Passages: {len(drug['passages'])}")
            if drug['indications']:
                ind_preview = drug['indications'][:150] + "..." if len(drug['indications']) > 150 else drug['indications']
                print(f"   🏥 Показання: {ind_preview}")
    else:
        print("   ❌ Результатів не знайдено")

print(f"\n✅ ФІНАЛЬНЕ ТЕСТУВАННЯ ЗАВЕРШЕНО!")
print(f"📊 Статистика:")
print(f"   - Тестових препаратів: {len(test_df)}")  
print(f"   - Passages створено: {len(assistant.passages)}")
print(f"   - Середня довжина passage: {sum(len(p) for p in assistant.passages) / len(assistant.passages):.0f} символів")

