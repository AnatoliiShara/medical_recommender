import sys
import pandas as pd
sys.path.append('src')

from search.enhanced_medical_assistant import EnhancedMedicalAssistant, SearchConfig

print("=== ТОЧКОВЕ НАЛАШТУВАННЯ СИСТЕМИ ===")

# Тестовий датасет з правильними препаратами
df = pd.read_parquet("/home/anatolii-shara/Documents/scraping_compendium/compendium_all.parquet")

# Фокус на проблемних запитах
problem_queries = {
    "цукровий діабет": ["діабет", "глікемі", "інсулін", "глюкоз"],
    "головний біль": ["головн біл", "мігрен", "цефалгі", "болю голов"]
}

selected_indices = []
for query, terms in problem_queries.items():
    for term in terms:
        count = 0
        for idx, row in df.iterrows():
            if count >= 10:
                break
            indications = str(row.get('Показання', '')).lower()
            if term in indications and idx not in selected_indices:
                selected_indices.append(idx)
                count += 1

test_df = df.iloc[selected_indices].copy()
print(f"Цільовий датасет: {len(test_df)} препаратів")

# Тест з мінімальними chunks для уникнення token warning
assistant = EnhancedMedicalAssistant()
assistant.build_from_dataframe(
    test_df, 
    encoder_model=None,
    medical_chunking=True,
    max_chunk_tokens=80  # Ще менше
)

print(f"Passages створено: {len(assistant.passages)}")

# Тестуємо проблемні запити
for query in ["цукровий діабет", "головний біль"]:
    print(f"\n🔍 ЗАПИТ: '{query}'")
    results = assistant.search(query, SearchConfig(show=3))
    
    for i, drug in enumerate(results):
        print(f"{i+1}. {drug['drug_name']} (Score: {drug['best_score']:.3f})")
        print(f"   Показання: {drug['indications'][:100]}...")

