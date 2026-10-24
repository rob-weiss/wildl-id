# Contributing

## Setup

Use Python 3.14.9 by default and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
```

Python 3.10 remains supported and is tested in CI. To select it locally, run `uv sync --locked --python 3.10`. Python 3.15 is not yet supported because PyTorch does not publish a `cp315` wheel.

Open this repository as the VS Code workspace and select `.venv` as the Python interpreter. `pyrightconfig.json` configures the `src/` import path and Python version for Pylance.

Camera photographs are not included in the repository. Keep source images out of commits; use temporary fixtures for tests.

## Checks

Run the same checks as CI before submitting a change:

```bash
uv lock --check
uv run --locked python -m compileall -q src wildl_id.py tests
PYTHONPATH=. uv run --locked python -m unittest discover -s tests
```

Keep documentation and user-facing prose in British English. Preserve established CLI flags and data formats unless a change requires otherwise. Changes to model processing must include regression tests that preserve existing behaviour.

## Data

The tracked labelled CSV is licensed under [CC BY 4.0](data-licence.md). This licence does not cover third-party model software or weights. Do not add source photographs without confirming their rights and updating the documentation.

[Home](index.md)
