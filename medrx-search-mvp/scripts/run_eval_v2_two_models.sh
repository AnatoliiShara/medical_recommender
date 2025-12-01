#!/usr/bin/env bash
set -euo pipefail

TAG="${1:?tag required (baseline|finetuned)}"
QUERIES="${2:?queries jsonl required}"
EMBED_MODEL="${3:?embed_model required}"
INDEX_DIR="${4:?index_dir required}"
FAISS_INDEX="${5:?faiss_index required}"
OUT_DIR="${6:?out_dir required}"

# optional passthrough (you can add more flags after OUT_DIR)
shift 6
EXTRA_ARGS=("$@")

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "[RUN] tag=$TAG"
echo "      queries=$QUERIES"
echo "      embed_model=$EMBED_MODEL"
echo "      index_dir=$INDEX_DIR"
echo "      faiss_index=$FAISS_INDEX"
echo "      out_dir=$OUT_DIR"
echo "============================================================"

python src/search/assistant_from_parquet.py \
  --index_dir "$INDEX_DIR" \
  --doc_index_dir "$INDEX_DIR" \
  --faiss_index "$FAISS_INDEX" \
  --embed_model "$EMBED_MODEL" \
  --queries "$QUERIES" \
  --dump_eval_dir "$OUT_DIR" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$OUT_DIR/run.log"

# HOTFIX: якщо код все ще пише в data/eval/predictions.jsonl — переносимо в OUT_DIR.
if [ -f "data/eval/predictions.jsonl" ]; then
  mv -f "data/eval/predictions.jsonl" "$OUT_DIR/predictions.jsonl"
fi

test -s "$OUT_DIR/predictions.jsonl" || { echo "[FAIL] predictions.jsonl not found in $OUT_DIR"; exit 2; }

echo "[OK] predictions: $(wc -l < "$OUT_DIR/predictions.jsonl") rows -> $OUT_DIR/predictions.jsonl"

python scripts/score_eval_jsonl.py \
  --queries "$QUERIES" \
  --predictions "$OUT_DIR/predictions.jsonl" \
  --out "$OUT_DIR/metrics.json" \
  --k 1 3 5 10 20 60 \
  2>&1 | tee "$OUT_DIR/metrics.txt"

echo "[OK] metrics saved: $OUT_DIR/metrics.json  (and metrics.txt)"
