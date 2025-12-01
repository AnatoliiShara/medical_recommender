"""
04_build_training_pairs.py

Створення training pairs для Stage 1 fine-tuning E5:
  - query (пацієнтський запит)
  - positive (текст препарату)
  - hard_negatives (список текстів інших препаратів)

Особливості:
  - Використовуємо тільки sampled препарати:
      data/training/finetuning/compendium_sampled.parquet
  - Використовуємо згенеровані запити:
      data/training/finetuning/queries_generated.jsonl
  - Документи та групи прив'язуються до drug_id.
  - Hard negatives:
      * частина з тієї ж фармгрупи (very hard),
      * частина з інших груп (medium hard).
  - Є tqdm progressbar і підтримка resume:
      * якщо training_pairs_stage1.jsonl уже існує,
        пропускаємо query з уже наявними qid.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set

import pandas as pd
from tqdm import tqdm

# -----------------------------
# Константи та конфіг
# -----------------------------
RANDOM_SEED = 42
NUM_NEGATIVES = 5

# Макс. довжини полів у документі (символи)
MAX_GROUP_CHARS = 200
MAX_INDICATION_CHARS = 600
MAX_COMPOSITION_CHARS = 200


def get_paths() -> Tuple[Path, Path, Path]:
    """
    Вираховуємо шляхи відносно кореня репозиторію.
    """
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]

    queries_path = (
        repo_root / "data" / "training" / "finetuning" / "queries_generated.jsonl"
    )
    compendium_path = (
        repo_root / "data" / "training" / "finetuning" / "compendium_sampled.parquet"
    )
    output_path = (
        repo_root / "data" / "training" / "finetuning" / "training_pairs_stage1.jsonl"
    )

    return queries_path, compendium_path, output_path


# -----------------------------
# Завантаження та підготовка
# -----------------------------
def load_queries(queries_path: Path) -> List[Dict[str, Any]]:
    """
    Завантажуємо згенеровані запити.

    Фільтруємо:
      - тільки записи з num_queries > 0,
      - пусті / пробільні рядки,
      - дублікати запитів для одного препарату.

    Додаємо поле qid — глобальний індекс query, який будемо
    використовувати для resume.
    """
    print(f"📂 Завантажуємо queries з: {queries_path}")

    raw_queries: List[Dict[str, Any]] = []

    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            if data.get("num_queries", 0) <= 0:
                continue

            drug_id = int(data["drug_id"])
            drug_name = str(data.get("drug_name", "")).strip()

            seen_for_drug: Set[str] = set()
            for q in data.get("queries", []):
                q = (q or "").strip()
                if not q:
                    continue
                if q in seen_for_drug:
                    continue
                seen_for_drug.add(q)

                raw_queries.append(
                    {
                        "query": q,
                        "drug_id": drug_id,
                        "drug_name": drug_name,
                    }
                )

    # Присвоюємо qid
    queries_with_id: List[Dict[str, Any]] = []
    for qid, item in enumerate(raw_queries):
        item = dict(item)
        item["qid"] = qid
        queries_with_id.append(item)

    print(f"   ✅ Валідних query-рядків: {len(queries_with_id):,}")
    return queries_with_id


def prepare_compendium(compendium_path: Path) -> pd.DataFrame:
    """
    Завантажуємо compendium_sampled.parquet і гарантуємо наявність колонки drug_id.
    Нормалізуємо фармакотерапевтичну групу.
    """
    print(f"\n📂 Завантажуємо sampled compendium: {compendium_path}")
    df = pd.read_parquet(compendium_path)

    # Якщо нема drug_id – створюємо з індексу (як у скрипті 03)
    if "drug_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["drug_id"] = df.index
    else:
        df = df.reset_index(drop=True)

    print(f"   Препаратів у sampled наборі: {len(df):,}")

    # Нормалізація груп
    df["group"] = (
        df["Фармакотерапевтична група"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    return df


def create_drug_documents(df: pd.DataFrame) -> Dict[int, str]:
    """
    Створює текстовий document для кожного препарату.
    Ключ – drug_id.
    """
    print("\n📝 Створення текстових document'ів для препаратів...")

    documents: Dict[int, str] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building documents"):
        drug_id = int(row["drug_id"])

        parts: List[str] = []

        # Назва препарату
        name = str(row.get("Назва препарату", "")).strip()
        if name:
            parts.append(name)

        # Фармакотерапевтична група (обрізаємо)
        group = str(row.get("Фармакотерапевтична група", "")).strip()
        if group:
            parts.append(group[:MAX_GROUP_CHARS])

        # Показання (основне поле для семантики)
        indications = str(row.get("Показання", "")).strip()
        if indications:
            parts.append(indications[:MAX_INDICATION_CHARS])

        # Склад (коротко)
        composition = str(row.get("Склад", "")).strip()
        if composition:
            parts.append(composition[:MAX_COMPOSITION_CHARS])

        doc_text = " | ".join(parts).strip()
        if not doc_text:
            continue

        documents[drug_id] = doc_text

    print(f"   ✅ Створено document'ів: {len(documents):,}")
    return documents


def build_group_indices(
    df: pd.DataFrame,
) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
    """
    Повертає:
      - groups_map: group -> [drug_id, ...]
      - id_to_group: drug_id -> group
    """
    print("\n📊 Побудова індексів груп...")

    groups_map: Dict[str, List[int]] = defaultdict(list)
    id_to_group: Dict[int, str] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing groups"):
        drug_id = int(row["drug_id"])
        group = str(row["group"])
        groups_map[group].append(drug_id)
        id_to_group[drug_id] = group

    print(f"   Груп покрито: {len(groups_map):,}")
    return groups_map, id_to_group


# -----------------------------
# Resume: читання вже готових qid
# -----------------------------
def load_processed_qids(output_path: Path) -> Set[int]:
    """
    Читає існуючий файл training_pairs_stage1.jsonl і повертає множину qid,
    які вже були оброблені.

    Якщо qid відсутній у metadata (старий формат) – просто ігноруємо та не
    використовуємо його для resume.
    """
    processed: Set[int] = set()

    if not output_path.exists():
        return processed

    print(f"\n📄 Знайдено існуючий файл training pairs: {output_path}")
    print("   Читаємо вже оброблені qid для resume...")

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            meta = obj.get("metadata", {})
            qid = meta.get("qid")
            if isinstance(qid, int):
                processed.add(qid)

    print(f"   Вже опрацьовано query з qid: {len(processed):,}")
    return processed


# -----------------------------
# Hard negatives mining + streaming запис
# -----------------------------
def mine_hard_negatives_and_write(
    queries_data: List[Dict[str, Any]],
    documents: Dict[int, str],
    groups_map: Dict[str, List[int]],
    id_to_group: Dict[int, str],
    output_path: Path,
    processed_qids: Set[int],
    num_negatives: int = NUM_NEGATIVES,
) -> Tuple[int, float, Dict[str, Any]]:
    """
    Створюємо training pairs з hard negatives і одразу записуємо їх у файл
    (streaming).

    Стратегія:
      - 0–3 негативи з тієї ж групи (якщо є достатньо варіантів),
      - решта – випадкові препарати з інших груп.

    Повертає:
      - total_pairs: скільки пар створено в цій сесії,
      - avg_negs: середня кількість негативів,
      - sample_pair: одна з отриманих пар (для логів).
    """
    print("\n🔨 Mining hard negatives та запис у файл...")

    random.seed(RANDOM_SEED)

    all_drug_ids = list(documents.keys())
    total_pairs = 0
    total_negs = 0
    sample_pair: Dict[str, Any] | None = None

    # Відкриваємо файл у режимі дописування (append)
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as f_out:
        iterator = tqdm(
            queries_data,
            desc="Building pairs",
            total=len(queries_data),
        )

        for item in iterator:
            qid = int(item["qid"])
            if qid in processed_qids:
                # Уже є в файлі – пропускаємо
                continue

            query = item["query"]
            pos_drug_id = int(item["drug_id"])
            drug_name = item["drug_name"]

            if pos_drug_id not in documents:
                continue

            positive_doc = documents[pos_drug_id]
            drug_group = id_to_group.get(pos_drug_id, "Unknown")

            neg_ids: List[int] = []

            # --- 1) Very hard negatives: з тієї ж групи ---
            same_group_ids = [
                d for d in groups_map.get(drug_group, []) if d != pos_drug_id
            ]
            if same_group_ids:
                n_same = min(3, num_negatives, len(same_group_ids))
                neg_ids.extend(random.sample(same_group_ids, n_same))

            # --- 2) Medium hard: випадкові з інших груп ---
            max_candidates = len(all_drug_ids) - 1  # окрім positive
            while len(neg_ids) < num_negatives and len(neg_ids) < max_candidates:
                candidate = random.choice(all_drug_ids)
                if candidate == pos_drug_id or candidate in neg_ids:
                    continue
                neg_ids.append(candidate)

            negative_docs = [documents[d_id] for d_id in neg_ids if d_id in documents]
            if not negative_docs:
                continue

            pair: Dict[str, Any] = {
                "query": query,
                "positive": positive_doc,
                "hard_negatives": negative_docs,
                "metadata": {
                    "qid": qid,
                    "drug_id": pos_drug_id,
                    "drug_name": drug_name,
                    "therapeutic_group": drug_group,
                    "num_negatives": len(negative_docs),
                },
            }

            # Записуємо в файл
            f_out.write(json.dumps(pair, ensure_ascii=False) + "\n")

            # Оновлюємо статистику
            total_pairs += 1
            total_negs += len(negative_docs)
            if sample_pair is None:
                sample_pair = pair

    if total_pairs == 0:
        return 0, 0.0, {}

    avg_negs = total_negs / total_pairs
    return total_pairs, avg_negs, sample_pair or {}


# -----------------------------
# main
# -----------------------------
def main():
    print("=" * 80)
    print("🔨 СТВОРЕННЯ TRAINING PAIRS З HARD NEGATIVES (Stage 1)")
    print("=" * 80)

    queries_path, compendium_path, output_path = get_paths()

    # 1) Перевірка наявності файлів
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file не знайдено: {queries_path}")
    if not compendium_path.exists():
        raise FileNotFoundError(f"Compendium file не знайдено: {compendium_path}")

    # 2) Завантаження даних
    df = prepare_compendium(compendium_path)
    queries_data = load_queries(queries_path)

    # 3) Створення документів та групових індексів
    documents = create_drug_documents(df)
    groups_map, id_to_group = build_group_indices(df)

    # 4) Resume: читаємо вже опрацьовані qid
    processed_qids = load_processed_qids(output_path)

    # 5) Mining + запис у файл
    total_pairs_new, avg_negs, sample_pair = mine_hard_negatives_and_write(
        queries_data=queries_data,
        documents=documents,
        groups_map=groups_map,
        id_to_group=id_to_group,
        output_path=output_path,
        processed_qids=processed_qids,
        num_negatives=NUM_NEGATIVES,
    )

    # Загальна кількість пар (старі + нові)
    total_pairs_total = len(processed_qids) + total_pairs_new

    print("\n📊 Статистика:")
    print(f"   Нових training pairs у цій сесії: {total_pairs_new:,}")
    print(f"   Загальна кількість пар (з урахуванням попередніх): {total_pairs_total:,}")
    if total_pairs_new > 0:
        print(f"   Середня кількість негативів на query (у нових парах): {avg_negs:.2f}")

    if total_pairs_new > 0 and sample_pair:
        print("\n📝 Приклад training pair (з цієї сесії):")
        print(f"   Query:    {sample_pair['query'][:120]}...")
        print(f"   Positive: {sample_pair['positive'][:120]}...")
        print(f"   Hard negatives: {len(sample_pair['hard_negatives'])}")

    print("\n" + "=" * 80)
    print("✅ TRAINING PAIRS ДЛЯ STAGE 1 ГОТОВІ (або оновлені)!")
    print("=" * 80)
    print(f"\n🎯 NEXT STEP: Fine-tuning E5-base на training_pairs_stage1.jsonl")


if __name__ == "__main__":
    main()
