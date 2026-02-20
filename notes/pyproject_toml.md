# pyproject.toml

## What is pyproject.toml?

The **single configuration file** for a modern Python project. Before it existed, you needed multiple files for different tools (`setup.py`, `setup.cfg`, `requirements.txt`, `.flake8`, `pytest.ini`, etc.).

```
Before (5+ files)          After (1 file)
├── setup.py               ├── pyproject.toml
├── setup.cfg
├── requirements.txt
├── .flake8
├── pytest.ini
```

Standardized by [PEP 621](https://peps.python.org/pep-0621/) (metadata) and [PEP 517](https://peps.python.org/pep-0517/) (build system).

## Three roles

1. **Package definition** — name, version, dependencies, entry points
2. **Build system** — how to build the package (e.g. setuptools)
3. **Tool configuration** — single place for ruff, pytest, mypy, etc.

## Section-by-section breakdown

### `[project]` — Package metadata

```toml
[project]
name = "deepfm"
version = "0.1.0"
description = "Production-grade DeepFM and variants for CTR prediction"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "pyyaml>=6.0",
    "dacite>=1.8",
]
```

- `name` — package name used by pip/uv
- `version` — semantic versioning
- `requires-python` — minimum Python version
- `dependencies` — core runtime deps, installed automatically with `pip install`

### `[project.optional-dependencies]` — Dev-only deps

```toml
[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov", "ruff"]
```

Only installed when you explicitly request them:

```bash
pip install -e ".[dev]"     # installs core + dev deps
pip install -e .            # installs core deps only
```

- `pytest` — test runner
- `pytest-cov` — coverage reporting
- `ruff` — linter and formatter (replaces flake8 + black + isort)

### What does `pip install -e ".[dev]"` mean?

Two parts:

- **`-e .`** (editable install) — installs the package from the current directory in editable mode. Creates a link to your source code instead of copying files into `site-packages`. Any code changes are immediately available without reinstalling.
- **`[dev]`** (optional extras) — also install the dev dependencies listed above.

### `[project.scripts]` — CLI entry point

```toml
[project.scripts]
deepfm = "deepfm.cli:main"
```

Creates a command-line executable when the package is installed:

- **`deepfm`** (left side) — the command name you type in terminal
- **`deepfm.cli:main`** (right side) — `module.path:function_name` to call

After install, pip creates a wrapper script in your Python bin directory that does:

```python
from deepfm.cli import main
main()
```

Three equivalent ways to run:

```bash
deepfm train --config ...            # installed script (from [project.scripts])
python -m deepfm train --config ...  # module entry (from __main__.py)
python -c "from deepfm.cli import main; main()"  # direct call
```

### `[tool.setuptools.packages.find]` — Package discovery

```toml
[tool.setuptools.packages.find]
include = ["deepfm*"]
```

Tells setuptools to **only** package directories matching `deepfm*`. The `*` wildcard catches sub-packages (`deepfm.data`, `deepfm.models.layers`, etc.).

**How `pip install -e .` finds the package:**

1. pip reads `pyproject.toml`, finds the build backend is `setuptools`
2. setuptools scans for directories containing `__init__.py`
3. The `include` filter narrows it to `deepfm*` only

Without the filter, setuptools auto-discovers **every** directory with `__init__.py` — including `tests/`, which we don't want installed as a package.

**Key signal:** a directory needs `__init__.py` to be recognized as a Python package.

### `[build-system]` — How to build

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"
```

Required by PEP 517. Tells pip/uv which tool to use when building from source.

### `[tool.ruff]` — Linter config

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

- `line-length = 100` — max line width (relaxed from default 88)
- `target-version` — lint for Python 3.10 syntax
- Rule sets:
  - **E** — pycodestyle errors (style issues)
  - **F** — pyflakes (unused imports, undefined names)
  - **I** — isort (import ordering)
  - **W** — pycodestyle warnings

### `[tool.pytest.ini_options]` — Test config

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests that require real ML-100K data (deselect with '-m \"not slow\"')",
]
```

- `testpaths` — where pytest looks for test files
- `markers` — custom markers so you can filter tests:

```bash
pytest -m "not slow"   # fast unit tests only
pytest                 # all tests including integration
```
