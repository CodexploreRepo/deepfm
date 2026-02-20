.PHONY: install train test lint format

VENV := .venv
UV := python3 -m uv
PYTHON := $(VENV)/bin/python

install:
	$(UV) venv $(VENV)
	$(UV) pip install -e ".[dev]" --python $(PYTHON)

train:
	$(PYTHON) -m deepfm train --config configs/deepfm_movielens.yaml

test:
	$(VENV)/bin/pytest tests/ -v

lint:
	$(VENV)/bin/ruff check deepfm/ tests/
	$(VENV)/bin/ruff format --check deepfm/ tests/

format:
	$(VENV)/bin/ruff check --fix deepfm/ tests/
	$(VENV)/bin/ruff format deepfm/ tests/
