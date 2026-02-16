UV := python3 -m uv
PYTHON := python3

.PHONY: install train test lint format clean

install:
	$(UV) venv && $(UV) pip install -e ".[dev]"

train:
	$(PYTHON) -m deepfm train --config configs/deepfm_movielens.yaml

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check deepfm/ tests/
	$(PYTHON) -m ruff format --check deepfm/ tests/

format:
	$(PYTHON) -m ruff check --fix deepfm/ tests/
	$(PYTHON) -m ruff format deepfm/ tests/

clean:
	rm -rf outputs/ __pycache__ .pytest_cache .ruff_cache *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
