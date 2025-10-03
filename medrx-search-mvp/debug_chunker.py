import sys
sys.path.append('src')

from preprocessing.medical_chunker import MedicalChunker

print("=== ДІАГНОСТИКА MEDICAL CHUNKER ===")

chunker = MedicalChunker()
test_text = "Показання: артеріальна гіпертензія. Протипоказання: вагітність."

print(f"Тестовий текст: '{test_text}'")
print(f"Довжина тексту: {len(test_text)} символів")

# Тест з різними min_chunk_chars
for min_chars in [10, 30, 60, 100]:
    chunks = chunker.smart_chunking(test_text, min_chunk_chars=min_chars)
    print(f"\nmin_chunk_chars={min_chars}: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  {i}: '{chunk}' ({len(chunk)} символів)")

# Тест з довшим текстом
long_text = """
Показання до застосування: артеріальна гіпертензія легкого та помірного ступеня, серцева недостатність хронічна, стенокардія напруги стабільна, профілактика інфаркту міокарда.
Протипоказання: діти до 18 років, вагітність та період лактації, гостра серцева недостатність, печінкова недостатність тяжкого ступеня, ниркова недостатність.
Дозування: по 1-2 таблетці двічі на день після їжі, курс лікування визначається лікарем індивідуально.
"""

print(f"\n=== ТЕСТ З ДОВШИМ ТЕКСТОМ ({len(long_text)} символів) ===")
long_chunks = chunker.smart_chunking(long_text, min_chunk_chars=60)
print(f"Chunks створено: {len(long_chunks)}")
for i, chunk in enumerate(long_chunks):
    print(f"{i}: '{chunk[:100]}...' ({len(chunk)} символів)")

