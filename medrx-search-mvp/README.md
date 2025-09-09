# Medical Search MVP

Розумна система пошуку медичних препаратів з використанням fine-tuned sentence transformers для української мови.

## Про проект

Цей проект демонструє створення медичного пошуковика з використанням:
- Fine-tuned sentence transformers для української медичної термінології
- Semantic search по 5000+ медичних препаратів
- Multiple Negative Ranking Loss для покращення релевантності
- Comprehensive EDA pipeline для медичних даних

## Структура проекту
src/
├── eda/              # Exploratory Data Analysis
├── prepare/          # Підготовка даних та training pairs
├── models/           # Fine-tuning sentence transformers
├── search/           # Пошукова система
└── api/              # REST API
data/                 # Дані (не включені в git через розмір)
models/               # Trained models (не включені в git)
configs/              # Конфігураційні файли

## Встановлення

1. Клонувати репозиторій:
```bash
git clone https://github.com/AnatoliiShara/medical_recommender.git
cd medical_recommender

## Створити virtual environment:
python -m venv venv
source venv/bin/activate  # Linux/Mac
# або venv\Scripts\activate  # Windows

## Встановити залежності:
pip install -r requirements.txt

# Як запустити
# 1. EDA (Exploratory Data Analysis)
python src/eda/comprehensive_medical_eda.py --dataset path/to/your/data.parquet

# 2. Підготовка training pairs
python src/prepare/improved_training_pairs.py --target 10000

# 3. Fine-tuning sentence transformer
# Тестовий запуск (100 прикладів, ~5 хвилин)
python src/models/finetune_fixed.py --test_run

# Повне навчання (~2-3 години на CPU)
python src/models/finetune_fixed.py

# 4.Тестування пошуку
python src/search/medical_search_prototype.py

# 5. API сервер (опціонально)
pip install fastapi uvicorn
python src/api/medical_search_api.py
# Відкрити http://localhost:8000/docs

### Дані
Проект використовує дані з compendium.com.ua (українська медична база даних).
Через обмеження GitHub, великі файли не включені в репозиторій:

Вихідні дані: ~228MB
Trained models: ~480MB каждая
Training pairs: ~87MB