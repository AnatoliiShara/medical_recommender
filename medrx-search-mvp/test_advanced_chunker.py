import sys
sys.path.append('src')

from preprocessing.medical_chunker import MedicalChunker

# Ініціалізація з tokenizer для точного chunking
chunker = MedicalChunker(tokenizer_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

test_text = '''
Показання до застосування: артеріальна гіпертензія легкого та помірного ступеня, серцева недостатність, стенокардія напруги. 
Протипоказання: діти до 18 років, вагітність, період лактації, гостра серцева недостатність, печінкова недостатність тяжкого ступеня.
Дозування та спосіб застосування: по 1 таблетці двічі на день після їжі, курс лікування визначається лікарем.
'''

print("=== ТЕСТ ВАШОГО ADVANCED CHUNKER ===")
chunks = chunker.smart_chunking(test_text, max_tokens=128, overlap_tokens=16)
for i, chunk in enumerate(chunks):
    print(f"{i}: \"{chunk}\"")
    print(f"   Довжина: {len(chunk)} символів")
    if chunker._tokenizer:
        tokens = len(chunker._tokenizer.encode(chunk))
        print(f"   Токенів: {tokens}")
    print()
