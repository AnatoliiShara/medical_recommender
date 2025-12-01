"""
02_sample_drugs_for_training.py

Розумний stratified sampling препаратів для генерації training data.

Цілі:
- Покрити різні фармакотерапевтичні групи (diversity).
- Жорстко тримати загальну кількість препаратів у діапазоні [target_min, target_max].
- Акуратно поводитися з порожніми / відсутніми фармакотерапевтичними групами.
- Оцінити потенціал для подальшої генерації hard negatives (але НЕ генерувати їх тут).

Результат:
- data/training/finetuning/compendium_sampled.parquet
- data/training/finetuning/sampling_stats.json
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json


# -----------------------------
# Константи для sampling
# -----------------------------
TARGET_MIN = 5000       # Мінімальна кількість препаратів у семплі
TARGET_MAX = 7000       # Максимальна кількість препаратів у семплі
SMALL_GROUP_THRESHOLD = 10   # Групи з ≤ 10 препаратів беремо повністю
MIN_PER_GROUP = 5            # Мінімум препаратів для великої групи

QUERIES_PER_DRUG = 7         # План: 7 запитів на препарат
HARD_NEG_PER_QUERY = 5       # План: до 5 hard negatives на запит (буде в наступному етапі)


def get_paths():
    """
    Обчислюємо шляхи відносно кореня репозиторію.
    Це дозволяє запускати скрипт з будь-якого поточного каталогу.
    """
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]  # .../medrx-search-mvp/

    data_raw_path = repo_root / "data" / "raw" / "compendium_all.parquet"
    training_dir = repo_root / "data" / "training" / "finetuning"
    training_dir.mkdir(parents=True, exist_ok=True)

    sampled_path = training_dir / "compendium_sampled.parquet"
    stats_path = training_dir / "sampling_stats.json"

    return data_raw_path, sampled_path, stats_path


def normalize_group_column(df: pd.DataFrame) -> pd.Series:
    """
    Нормалізуємо колонку 'Фармакотерапевтична група':
    - NaN → ''
    - strip() пробіли
    - порожні строки → NaN
    - NaN → 'Unknown'
    """
    col = df["Фармакотерапевтична група"].fillna("").astype(str).str.strip()
    col = col.replace("", np.nan).fillna("Unknown")
    return col


def stratified_sample(
    df_valid: pd.DataFrame,
    target_min: int,
    target_max: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Stratified sampling за фармакотерапевтичними групами з жорстким контролем
    діапазону [target_min, target_max].

    Алгоритм:
      1. Рахуємо базову частку base_fraction = target_max / len(df_valid).
      2. Для малих груп (≤ SMALL_GROUP_THRESHOLD) беремо всі.
      3. Для великих груп беремо ~count * base_fraction, але не менше MIN_PER_GROUP.
      4. Після первинного stratified sampling:
         - якщо > target_max → випадково даунсемплимо до target_max;
         - якщо < target_min → добираємо випадкові препарати з решти df_valid
           до target_min.
    """
    group_counts = df_valid["group"].value_counts()
    total_valid = len(df_valid)
    base_fraction = target_max / total_valid

    print(f"\n📊 Всього валідних препаратів: {total_valid:,}")
    print(f"📊 Унікальних терапевтичних груп: {len(group_counts):,}")
    print(f"⚖️  Базова частка для великих груп: {base_fraction:.3f}")

    sampled_indices = []

    for group, count in group_counts.items():
        group_df = df_valid[df_valid["group"] == group]

        if count <= SMALL_GROUP_THRESHOLD:
            # Малі групи — беремо повністю
            sample_size = count
        else:
            # Великі групи — пропорційний sampling
            sample_size = int(round(count * base_fraction))
            sample_size = max(MIN_PER_GROUP, min(sample_size, count))

        sample = group_df.sample(
            n=sample_size,
            random_state=random_state,
            replace=False,
        )
        sampled_indices.extend(sample.index.tolist())

    # Первинний семпл
    df_sampled = df_valid.loc[sampled_indices].copy()

    # Прибираємо дублікати за назвою препарату (на всяк випадок)
    df_sampled = df_sampled.drop_duplicates(subset=["Назва препарату"])

    print(f"\n🔁 Після первинного stratified sampling:")
    print(f"   Кількість препаратів: {len(df_sampled):,}")

    # Жорстко забезпечуємо діапазон [target_min, target_max]
    if len(df_sampled) > target_max:
        # Даунсемплимо до target_max
        df_sampled = df_sampled.sample(
            n=target_max,
            random_state=random_state,
            replace=False,
        )
        print(f"   🔻 Даунсемпл до TARGET_MAX = {target_max:,}")
    elif len(df_sampled) < target_min:
        # Добираємо препарати з решти df_valid
        deficit = target_min - len(df_sampled)
        print(f"   🔺 Мало препаратів, добираємо ще: {deficit:,}")

        remaining = df_valid.drop(index=df_sampled.index)
        add_size = min(deficit, len(remaining))

        if add_size > 0:
            extra = remaining.sample(
                n=add_size,
                random_state=random_state,
                replace=False,
            )
            df_sampled = pd.concat([df_sampled, extra], axis=0)

    # Фінальна унікальність за назвою
    df_sampled = df_sampled.drop_duplicates(subset=["Назва препарату"])

    print(f"\n✅ Фінальний розмір семплу: {len(df_sampled):,} препаратів")
    return df_sampled


def sample_drugs_stratified():
    print("=" * 80)
    print("🎯 SAMPLING ПРЕПАРАТІВ ДЛЯ TRAINING (stratified за фармакогрупами)")
    print("=" * 80)

    data_raw_path, sampled_path, stats_path = get_paths()

    # 1. Завантаження full dataset
    print(f"\n📂 Завантажуємо Compendium з: {data_raw_path}")
    df = pd.read_parquet(data_raw_path)
    print(f"   Всього препаратів: {len(df):,}")

    # 2. Фільтр: тільки з показаннями
    df_valid = df[df["Показання"].notna()].copy()
    print(f"✅ Препаратів з непорожніми 'Показання': {len(df_valid):,}")

    # 3. Нормалізуємо фармакотерапевтичні групи
    df_valid["group"] = normalize_group_column(df_valid)

    # 4. Stratified sampling з жорстким діапазоном
    df_sampled = stratified_sample(
        df_valid=df_valid,
        target_min=TARGET_MIN,
        target_max=TARGET_MAX,
        random_state=42,
    )

    # 5. Quality checks
    avg_indications_len = df_sampled["Показання"].fillna("").str.len().mean()
    groups_covered = df_sampled["group"].nunique()

    print(f"\n🔍 Quality checks:")
    print(f"   Середня довжина 'Показання': {avg_indications_len:.0f} символів")
    print(f"   Покрито терапевтичних груп: {groups_covered:,}")

    # 6. Оцінка потенційного training data
    expected_queries = len(df_sampled) * QUERIES_PER_DRUG
    expected_positive_pairs = expected_queries
    expected_pairs_with_hn = expected_queries * (1 + HARD_NEG_PER_QUERY)

    print(f"\n📈 Потенціал для training data:")
    print(f"   🔹 Queries (≈{QUERIES_PER_DRUG} на препарат): {expected_queries:,}")
    print(f"   🔹 Позитивні пари (query–positive_passage): {expected_positive_pairs:,}")
    print(
        f"   🔹 Макс. потенціал з hard negatives "
        f"(+{HARD_NEG_PER_QUERY} на query): до {expected_pairs_with_hn:,} пар"
    )
    print("   ⚠️ Hard negatives БУДУТЬ згенеровані на наступному етапі "
          "(03_build_training_pairs.py). У цьому скрипті ми лише семплимо препарати.")

    # 7. Збереження семплу
    df_sampled.to_parquet(sampled_path)
    print(f"\n💾 Sampled dataset збережено: {sampled_path}")

    # 8. Збереження статистики
    stats = {
        "total_drugs": int(len(df)),
        "valid_drugs": int(len(df_valid)),
        "sampled_drugs": int(len(df_sampled)),
        "coverage_pct": float(len(df_sampled) / len(df_valid) * 100.0),
        "therapeutic_groups_total": int(df_valid["group"].nunique()),
        "therapeutic_groups_covered": int(groups_covered),
        "target_min": int(TARGET_MIN),
        "target_max": int(TARGET_MAX),
        "queries_per_drug": int(QUERIES_PER_DRUG),
        "hard_neg_per_query": int(HARD_NEG_PER_QUERY),
        "expected_queries": int(expected_queries),
        "expected_positive_pairs": int(expected_positive_pairs),
        "expected_pairs_with_hard_negatives": int(expected_pairs_with_hn),
    }

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"📊 Statistics збережено: {stats_path}")

    print("\n" + "=" * 80)
    print("✅ SAMPLING ЗАВЕРШЕНО")
    print("=" * 80)

    return df_sampled


if __name__ == "__main__":
    df_sampled = sample_drugs_stratified()

    print(f"\n🎯 NEXT STEP:")
    print(f"   Згенерувати пацієнтські запити для {len(df_sampled):,} препаратів")
    print(f"   (напр. через Gemini API) і побудувати training pairs для файнтюнінгу.")
