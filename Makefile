# KCAC OCR Pipeline — convenience targets
# Run `make help` to see available commands.

.PHONY: help install install-ocr install-calamari install-transformers-gpu \
        doctor smoke test lint typecheck

help:
	@echo "Setup targets:"
	@echo "  make install               Core deps (Python 3.13 compatible)"
	@echo "  make install-ocr           + Kraken OCR (requires Python 3.10/3.11 env)"
	@echo "  make install-calamari      Calamari-only env (separate from Kraken)"
	@echo "  make install-transformers  + Transformers GPU backend for Qwen"
	@echo ""
	@echo "Run targets:"
	@echo "  make doctor                Check environment and config"
	@echo "  make smoke                 One-page smoke test (bootstrap → reports)"
	@echo "  make test                  pytest + ruff + mypy"
	@echo "  make lint                  ruff check only"
	@echo "  make typecheck             mypy only"

install:
	pip install -r requirements.txt

install-ocr:
	pip install -r requirements.txt -r requirements-ocr-py310.txt

install-calamari:
	pip install -r requirements-calamari-py310.txt

install-transformers:
	pip install -r requirements-transformers-gpu.txt

doctor:
	python -m pipeline doctor

smoke:
	python -m pipeline bootstrap --limit 1
	python -m pipeline consensus --limit 1
	python -m pipeline pagexml --limit 1
	python -m pipeline escriptorium --limit 1
	python -m pipeline queue --limit 1
	python -m pipeline reports --limit 1

test:
	python -m pytest
	python -m ruff check .
	python -m mypy pipeline

lint:
	python -m ruff check .

typecheck:
	python -m mypy pipeline
