#!/usr/bin/env bash
set -euo pipefail

# ---------- НАЛАШТУВАННЯ ----------
BASE_CMD="python -m eval.run_eval_queries"

# 1) Індекс (Parquet) — ВКАЖИ ПРАВИЛЬНИЙ ШЛЯХ!
PARQUET="${PARQUET:-data/index/doc_chunks.parquet}"

# 2) Запити — можна кілька файлів
QUERIES_LIST=(
  "data/eval/queries_medrx_ua.enhanced.jsonl"
  "data/eval/queries_autogen_ua.enhanced.jsonl"
)

# 3) Aliases (необов’язково, але корисно)
ALIASES_CSV="${ALIASES_CSV:-data/dicts/brand_inn_aliases.PATCHED2.csv}"

# 4) Сітка top-K (еквівалент нашого doc_topN)
K_LIST=("600" "800" "1000")
# -----------------------------------

timestamp="$(date +%Y%m%d_%H%M%S)"
OUTDIR="runs/bm25_docpref_${timestamp}"
mkdir -p "$OUTDIR"

echo "[INFO] OUTDIR=$OUTDIR"
echo "[INFO] BASE_CMD=$BASE_CMD"

# sanity checks
if [[ ! -f "$PARQUET" ]]; then
  echo "[ERR] PARQUET не знайдено: $PARQUET
  Задай змінну середовища PARQUET або відредагуй скрипт.
  Приклад: PARQUET=data/index/medrx_docs.parquet scripts/run_bm25_docpref_grid.sh" >&2
  exit 1
fi

for K in "${K_LIST[@]}"; do
  echo "[RUN] k=${K}"
  for Q in "${QUERIES_LIST[@]}"; do
    if [[ ! -f "$Q" ]]; then
      echo "[WARN] пропускаю (queries не знайдено): $Q" >&2
      continue
    fi
    qname="$(basename "${Q%.*}")"
    run_dir="${OUTDIR}/k${K}/${qname}"
    mkdir -p "$run_dir"

    CMD=( $BASE_CMD --parquet "$PARQUET" --queries "$Q" --k "$K" )
    [[ -f "$ALIASES_CSV" ]] && CMD+=( --aliases "$ALIASES_CSV" )

    echo -n "[CMD] "; printf '%q ' "${CMD[@]}"; echo
    "${CMD[@]}" 2>&1 | tee "${run_dir}/stdout.log"

    # Витяг примітивних метрик зі stdout (скорегуй шаблони за потреби)
    grep -E 'nDCG@10|nDCG@5|MRR@10|Recall@100\b|Recall@1000\b|MAP@10' -n \
      "${run_dir}/stdout.log" > "${run_dir}/quick_metrics.txt" || true
  done
done

# короткий зведений дашборд по quick_metrics
python - <<'PY'
import os,glob,sys,re
root=sys.argv[1]
rows=[]
for path in glob.glob(os.path.join(root,"k*","*","quick_metrics.txt")):
    run = os.path.dirname(path).split(os.sep)[-2:]
    run = "/".join(run)  # k600/queries_medrx_ua.enhanced
    metrics={}
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            # шукаємо "key: value" або "key=value"
            m = re.findall(r'([A-Za-z@0-9]+)\s*[:=]\s*([0-9.]+)', line)
            for k,v in m:
                metrics[k]=v
    rows.append((run,metrics))
if rows:
    print("run,nDCG@10,nDCG@5,MRR@10,Recall@100,Recall@1000,MAP@10")
    for run, m in sorted(rows):
        print(",".join([
            run,
            m.get("nDCG@10",""),
            m.get("nDCG@5",""),
            m.get("MRR@10",""),
            m.get("Recall@100",""),
            m.get("Recall@1000",""),
            m.get("MAP@10",""),
        ]))
else:
    print("[WARN] quick_metrics.txt не знайдено — перевір stdout.log", file=sys.stderr)
PY "$OUTDIR" | tee "${OUTDIR}/summary.csv"
