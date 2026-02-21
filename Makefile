.PHONY: install train evaluate compare test lint format

VENV := .venv
UV := python3 -m uv
PYTHON := $(VENV)/bin/python

ARGS ?=
RUNS_DIR ?= outputs

install:
	$(UV) venv $(VENV)
	$(UV) pip install -e ".[dev]" --python $(PYTHON)

train:
	$(PYTHON) -m deepfm train --config configs/deepfm_movielens.yaml \
		$(if $(ARGS),--override $(ARGS),)

evaluate:
	$(PYTHON) -m deepfm evaluate --config configs/deepfm_movielens.yaml \
		$(if $(ARGS),--override $(ARGS),)

compare:
	$(PYTHON) -m deepfm compare --dir $(RUNS_DIR)

test:
	$(VENV)/bin/pytest tests/ -v

lint:
	$(VENV)/bin/ruff check deepfm/ tests/
	$(VENV)/bin/ruff format --check deepfm/ tests/

format:
	$(VENV)/bin/ruff check --fix deepfm/ tests/
	$(VENV)/bin/ruff format deepfm/ tests/
