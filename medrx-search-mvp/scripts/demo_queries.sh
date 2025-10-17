#!/usr/bin/env bash
set -euo pipefail
python3 -m src.search.assistant_from_parquet "$@"
