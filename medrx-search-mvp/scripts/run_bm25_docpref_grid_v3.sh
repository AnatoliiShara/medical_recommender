#!/usr/bin/env bash
set -euo pipefail

# --------- Параметри з оточення (є значення за замовчуванням) ----------
PARQUET="${PARQUET:-artifacts_backup/clean_medical.parquet}"
MODE="${MODE:-hybrid}"            # fast | hybrid | hybrid_ce
SPLITS="${SPLITS:-queries_medrx_ua.enhanced}"
K_VALUES="${K_VALUES:-600}"       # можна "600 800" тощо
OUTROOT="runs/bm25_docpref_v3_$(date +%Y%m%d_%H%M%S)"

echo "[INFO] OUTROOT=$OUTROOT"
echo "[INFO] PARQUET=$PARQUET"
echo "[INFO] MODE=$MODE"

mkdir -p "$OUTROOT"

# --------- Базова команда-модуль ---------
BASE_CMD=(python -m eval.evaluate_bench)
echo "[INFO] BASE_CMD=${BASE_CMD[*]}"

# --------- Налаштування режимів ----------
ENCODER=none
CE_MODEL=none
RERANK_TOP=0

case "$MODE" in
  fast)
    ENCODER=none
    CE_MODEL=none
    RERANK_TOP=0
    ;;
  hybrid)
    ENCODER=intfloat/multilingual-e5-small
    CE_MODEL=none
    RERANK_TOP=0
    ;;
  hybrid_ce)
    ENCODER=intfloat/multilingual-e5-small
    CE_MODEL=BAAI/bge-reranker-v2-m3
    RERANK_TOP=50
    ;;
  *)
    echo "[ERR] Unknown MODE='$MODE' (use: fast | hybrid | hybrid_ce)"; exit 1;;
esac

# --------- Запуски по сплітам та k ----------
for split in $SPLITS; do
  for k in $K_VALUES; do
    OUTDIR="$OUTROOT/k$k/$split"
    mkdir -p "$OUTDIR"
    echo "[RUN] split=$split k=$k -> $OUTDIR"

    # Команда як масив
    CMD=(
      "${BASE_CMD[@]}"
      --queries "data/eval/${split}.jsonl"
      --dataset "$PARQUET"
      --top_k "$k"
      --prefer_indications
      --encoder_model "$ENCODER"
      --reranker_model "$CE_MODEL" --rerank_top "$RERANK_TOP"
      --alpha 60 --ks 3 5 10
      --out_json "$OUTDIR/preds.json"
      --out_csv  "$OUTDIR/preds.csv"
    )

    # Додати product_map, якщо задано змінною оточення
    if [[ -n "${PRODUCT_MAP:-}" ]]; then
      CMD+=( --product_map "$PRODUCT_MAP" --warnings )
    fi

    # Лог: покажемо повну команду
if [[ -n "$RERANKER" ]]; then
  CMD+=( --reranker_model "$RERANKER" );
fi;
if [[ -n "$RERANK_TOP" ]]; then
  CMD+=( --rerank_top "$RERANK_TOP" );
fi
    printf '[CMD] ' | tee "$OUTDIR/stdout.log"
    printf '%q ' "${CMD[@]}" | tee -a "$OUTDIR/stdout.log"
    echo | tee -a "$OUTDIR/stdout.log"

    # Запуск з tee у stdout.log
    "${CMD[@]}" | tee -a "$OUTDIR/stdout.log"
  done
done
