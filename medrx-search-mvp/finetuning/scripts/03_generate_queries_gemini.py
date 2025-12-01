"""
03_generate_queries_gemini.py

Генерація пацієнтських запитів для sampled препаратів через Gemini.
Вхід:
  - data/training/finetuning/compendium_sampled.parquet

Вихід:
  - data/training/finetuning/queries_generated.jsonl
    Формат рядка:
    {
      "drug_id": int,
      "drug_name": str,
      "url": str | null,
      "therapeutic_group": str | null,
      "num_queries": int,
      "queries": [str, ...]
    }

Особливості:
  - tqdm progressbar по препаратах.
  - Лічильник API-викликів до Gemini.
  - Якщо отримаємо помилку, схожу на quota/rate-limit (429, quota, rate, exhausted),
    скрипт коректно зупиняється.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # якщо tqdm не встановлено, будемо без прогресбару

import google.generativeai as genai


# -----------------------------
# Конфіг
# -----------------------------
QUERIES_PER_DRUG = 7
MAX_INDICATION_CHARS = 1200  # обрізаємо надто довгі "Показання"
REQUEST_SLEEP_SEC = 1     # невелика пауза між запитами, щоб не душити API


def get_paths():
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]

    sampled_path = repo_root / "data" / "training" / "finetuning" / "compendium_sampled.parquet"
    output_path = repo_root / "data" / "training" / "finetuning" / "queries_generated.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return sampled_path, output_path


def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY не заданий в env")

    # За замовчуванням використовуємо швидку модель
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    return model, model_name


def build_prompt(drug_row: pd.Series, queries_per_drug: int) -> str:
    """
    Формуємо prompt для Gemini.
    Даємо назву, фармгрупу, показання, але просимо НЕ згадувати назву
    в запитах — тільки симптоми / ситуації.
    """
    name = str(drug_row.get("Назва препарату", "")).strip()
    group = str(drug_row.get("Фармакотерапевтична група", "")).strip()
    indications = str(drug_row.get("Показання", "")).strip()

    if len(indications) > MAX_INDICATION_CHARS:
        indications_short = indications[:MAX_INDICATION_CHARS] + "..."
    else:
        indications_short = indications

    prompt = f"""
Ти допомагаєш створювати псевдо-анонімізовані пошукові запити пацієнтів
для тренування медичної пошукової системи.

Є лікарський засіб з такими властивостями (НЕ згадуй його назву у відповідях):

Назва препарату: {name}
Фармакотерапевтична група: {group if group else "—"}
Показання (коротко, технічний опис з інструкції):
{indications_short}

ЗАВДАННЯ:
- Згенеруй {queries_per_drug} різних реалістичних пошукових запитів українською мовою,
  які міг би ввести пацієнт в онлайн-аптеці або чат-боті, шукаючи препарат
  з такими показаннями.
- Використовуй побутову мову: опис симптомів, скарг, ситуацій, тривалості,
  іноді згадуй вік ("дитина", "літня людина"), але НЕ використовуй справжню
  назву препарату.
- Не давай медичних рекомендацій або діагнозів, лише формулювання запитів.
- Кожен запит має бути окремим JSON-рядком у масиві.

ФОРМАТ ВІДПОВІДІ:
Поверни ЧИСТИЙ JSON такої структури БЕЗ додаткового тексту:

{{
  "queries": [
    "перший запит...",
    "другий запит...",
    "... і так далі"
  ]
}}
"""
    return prompt.strip()


def parse_queries_from_response(text: str) -> List[str]:
    """
    Пробуємо витягнути список queries з відповіді Gemini.
    Очікуємо JSON {"queries": [...]}.
    Якщо щось пішло не так — простий fallback: рядки, розбиті по лініях.
    """
    text = text.strip()

    # 1. Спроба як JSON
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start : end + 1]
            data = json.loads(json_str)
        else:
            data = json.loads(text)

        queries = data.get("queries") or data.get("query") or data
        if isinstance(queries, list):
            return [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    except Exception:
        pass

    # 2. Fallback: розбити по рядках / маркерах
    lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for prefix in ("-", "*", "•", "—", "–", "1.", "2.", "3.", "4.", "5.", "6.", "7."):
            if line.lower().startswith(prefix.lower()):
                line = line[len(prefix) :].strip()
        if line:
            lines.append(line)

    return lines


def generate_queries_for_drug(model, drug_row: pd.Series, queries_per_drug: int) -> List[str]:
    prompt = build_prompt(drug_row, queries_per_drug)
    response = model.generate_content(prompt)
    text = response.text or ""
    queries = parse_queries_from_response(text)

    if len(queries) > queries_per_drug:
        queries = queries[:queries_per_drug]

    return queries


def is_quota_error(exc: Exception) -> bool:
    """
    Евристика: визначити, чи схожа помилка на quota/rate-limit.
    Працюємо тільки зі строковим повідомленням — універсально для різних версій SDK.
    """
    msg = str(exc).lower()
    keywords = [
        "quota",
        "rate",
        "429",
        "resourceexhausted",
        "exceeded",
        "too many requests",
        "insufficient",
        "billing",
    ]
    return any(k in msg for k in keywords)


def main():
    sampled_path, output_path = get_paths()

    print("=" * 80)
    print("🧠 GENERATE QUERIES ДЛЯ SAMPLED ПРЕПАРАТІВ (Gemini)")
    print("=" * 80)
    print(f"\n📂 Завантажуємо sampled dataset: {sampled_path}")

    df = pd.read_parquet(sampled_path).reset_index(drop=True)
    df["drug_id"] = df.index
    total_drugs = len(df)

    print(f"   Препаратів у семплі: {total_drugs:,}")

    model, model_name = init_gemini()
    print(f"\n🤖 Використовуємо модель Gemini: {model_name}")
    print(f"   Запитів на препарат: {QUERIES_PER_DRUG}")
    print(f"   Вихідний файл: {output_path}")

    # Якщо файл вже існує — читаємо, щоб не дублювати (resume)
    existing_drug_ids = set()
    if output_path.exists():
        print("\n📄 Знайдено існуючий файл queries_generated.jsonl — читаємо вже оброблені drug_id")
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing_drug_ids.add(int(obj["drug_id"]))
                except Exception:
                    continue
        print(f"   Вже згенеровано для {len(existing_drug_ids):,} препаратів")

    # Лічильник API-викликів
    api_calls = 0
    quota_hit = False

    out_f = output_path.open("a", encoding="utf-8")

    # Вибираємо тільки ті рядки, які ще не оброблені
    df_to_process = df[~df["drug_id"].isin(existing_drug_ids)]

    iterator = df_to_process.iterrows()
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(df_to_process), desc="Generating queries")

    try:
        for _, row in iterator:
            drug_id = int(row["drug_id"])

            try:
                queries = generate_queries_for_drug(model, row, QUERIES_PER_DRUG)
                api_calls += 1
            except Exception as e:
                if is_quota_error(e):
                    print(f"\n⛔ Отримали помилку, схожу на quota/rate-limit для drug_id={drug_id}: {e}")
                    print("   Зупиняємо генерацію, щоб не перевищувати ліміти.")
                    quota_hit = True
                    break
                else:
                    print(f"\n⚠️ Помилка генерації для drug_id={drug_id}: {e}")
                    queries = []

            record: Dict[str, Any] = {
                "drug_id": drug_id,
                "drug_name": str(row.get("Назва препарату", "")).strip(),
                "url": str(row.get("url", "")).strip() or None,
                "therapeutic_group": str(row.get("Фармакотерапевтична група", "")).strip() or None,
                "num_queries": len(queries),
                "queries": queries,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            time.sleep(REQUEST_SLEEP_SEC)
    finally:
        out_f.close()

    print("\n✅ ГЕНЕРАЦІЯ ЗАПИТІВ ЗАВЕРШЕНА (або перервана через ліміт)")
    print(f"📊 Всього API-викликів до Gemini у цій сесії: {api_calls}")
    if quota_hit:
        print("⚠️ Скрипт зупинився через підозру на quota/rate-limit помилку.")
        print("   Після розширення лімітів або паузи можна перезапустити —")
        print("   скрипт продовжить з місця, де зупинився (resume).")

    print(f"📄 Поточний файл з результатами: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
