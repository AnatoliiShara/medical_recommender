#!/usr/bin/env bash
set -euo pipefail

OUTROOT=${OUTROOT:-"runs/bm25_docpref_v2_$(date +%Y%m%d_%H%M%S)"}
BASE_CMD="python -m eval.evaluate_bench"

# Дані
PARQUET=${PARQUET:-"artifacts_backup/clean_medical.parquet"}
SPLITS=${SPLITS:-"queries_medrx_ua.enhanced queries_autogen_ua.enhanced"}
K_VALUES=${K_VALUES:-"600"}   # можна: "600 800 1000"

# Моделі / ваги
ENCODER_MODEL=${ENCODER_MODEL:-"intfloat/multilingual-e5-base"}
RERANKER_MODEL=${RERANKER_MODEL:-"BAAI/bge-reranker-v2-m3"}
RERANK_TOP=${RERANK_TOP:-100}
ALPHA=${ALPHA:-60}            # RRF alpha
KS=${KS:-"1 3 5 10"}          # для метрик

echo "[INFO] OUTROOT=$OUTROOT"
echo "[INFO] BASE_CMD=$BASE_CMD"
echo "[INFO] PARQUET=$PARQUET"

for K in $K_VALUES; do
  echo "[RUN] k=$K"
  for SPLIT in $SPLITS; do
    QFILE="data/eval/${SPLIT}.jsonl"
    OUTDIR="$OUTROOT/k$K/$SPLIT"
    mkdir -p "$OUTDIR"

    # Якщо вже є preds.json — пропускаємо (ідемпотентність)
    if [[ -f "$OUTDIR/preds.json" ]]; then
      echo "[SKIP] $OUTDIR/preds.json існує"
      continue
    fi

    echo "[CMD] $BASE_CMD --queries $QFILE --dataset $PARQUET --top_k $K \\"
    echo "      --prefer_indications --encoder_model $ENCODER_MODEL \\"
    echo "      --reranker_model $RERANKER_MODEL --rerank_top $RERANK_TOP \\"
    echo "      --alpha $ALPHA --ks $KS --out_json $OUTDIR/preds.json \\"
    echo "      --out_csv $OUTDIR/preds.csv"

    $BASE_CMD \
      --queries "$QFILE" \
      --dataset "$PARQUET" \
      --top_k "$K" \
      --prefer_indications \
      --encoder_model "$ENCODER_MODEL" \
      --reranker_model "$RERANKER_MODEL" \
      --rerank_top "$RERANK_TOP" \
      --alpha "$ALPHA" \
      --ks $KS \
      --out_json "$OUTDIR/preds.json" \
      --out_csv "$OUTDIR/preds.csv" \
      |& tee "$OUTDIR/stdout.log"
  done
done

echo "[DONE] Усе збережено в $OUTROOT"
