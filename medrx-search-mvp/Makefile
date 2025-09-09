PY=python
VENV=.venv

.PHONY: venv install eda ingest index help

help:
\t@echo "make venv        - create venv"
\t@echo "make install     - install deps"
\t@echo "make ingest      - створити symlink на parquet у data/raw"
\t@echo "make eda         - запустити Jupyter"
\t@echo "make up-es       - підняти OpenSearch (якщо треба BM25)"
\t@echo "make index       - побудувати FAISS індекс (vector)"
\t@echo "make api         - запустити FastAPI"

venv:
\tpython3 -m venv $(VENV)
\t. $(VENV)/bin/activate; pip install --upgrade pip

install:
\t. $(VENV)/bin/activate; pip install -r requirements.txt

ingest:
\tbash scripts/ingest_symlink.sh

eda:
\t. $(VENV)/bin/activate; jupyter notebook

up-es:
\tbash scripts/up_es.sh

index:
\t. $(VENV)/bin/activate; $(PY) src/index/build_faiss.py

api:
\t. $(VENV)/bin/activate; uvicorn src.api.app:app --reload --port 8081
