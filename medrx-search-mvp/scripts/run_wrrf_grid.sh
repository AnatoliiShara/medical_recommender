#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=${PYTHONPATH:-src}   # корінь уже в sys.path, тому sitecustomize.py підхопиться авто
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
: "${OMP_NUM_THREADS:=2}"; : "${MKL_NUM_THREADS:=2}"

SPLIT=${SPLIT:-queries_autogen_ua.enhanced}  # або queries_medrx_ua.enhanced
ALPHA=${ALPHA:-15}
TOPK=${TOPK:-600}
ENCODER_MODEL=${ENCODER_MODEL:-intfloat/multilingual-e5-small}
DATASET=${DATASET:-artifacts_backup/clean_medical.parquet}
PRODUCT_MAP=${PRODUCT_MAP:-data/dicts/brand_inn_aliases.PATCHED2.csv}
WEIGHT_PAIRS=${WEIGHT_PAIRS:-"1,1 2,1 1,2"}

OUTROOT="runs/wrrf_env_a${ALPHA}_$(date +%Y%m%d_%H%M%S)"
echo "[INFO] OUTROOT=$OUTROOT"

for pair in ${WEIGHT_PAIRS}; do
  IFS=, read -r WB WD <<< "$pair"
  OUTDIR="$OUTROOT/k${TOPK}/${SPLIT}/w${WB}_${WD}"
  mkdir -p "$OUTDIR"
  echo "[RUN] ALPHA=$ALPHA WRRF_W_BM25=$WB WRRF_W_DENSE=$WD -> $OUTDIR"

  WRRF_ALPHA="$ALPHA" WRRF_W_BM25="$WB" WRRF_W_DENSE="$WD" \
  python -m eval.evaluate_bench \
    --queries "data/eval/${SPLIT}.jsonl" \
    --dataset "$DATASET" \
    --top_k "$TOPK" --prefer_indications \
    --encoder_model "$ENCODER_MODEL" \
    --reranker_model none --rerank_top 0 \
    --alpha "$ALPHA" \
    --ks 3 5 10 \
    --product_map "$PRODUCT_MAP" --warnings \
    --out_json "$OUTDIR/preds.json" \
    --out_csv  "$OUTDIR/preds.csv" \
  | tee "$OUTDIR/stdout.log"
done
