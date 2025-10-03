import sys
sys.path.append('src')

print("=== SMOKE TEST: Enhanced Medical Assistant ===")

try:
    from search.enhanced_medical_assistant import EnhancedMedicalAssistant, SearchConfig
    print("✅ Import successful")
    
    # Створюємо інстанс
    assistant = EnhancedMedicalAssistant()
    print("✅ Instance created")
    
    # Перевіряємо чи є MedicalChunker
    test_text = "Показання: артеріальна гіпертензія. Протипоказання: вагітність."
    chunks = assistant.medical_chunker.smart_chunking(test_text)
    print(f"✅ MedicalChunker works: {len(chunks)} chunks created")
    for i, chunk in enumerate(chunks):
        print(f"   {i}: {chunk}")
    
    print("✅ SMOKE TEST PASSED - готовий до тестування на даних")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

