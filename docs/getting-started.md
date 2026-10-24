# Getting Started

Wildlife Identification and Analysis provides Python tools for processing camera-trap images and analysing labelled detections.

## Requirements

Python 3.14.9 is the project default. Python 3.10 is also supported and tested in CI. Python 3.15 is not yet supported because PyTorch has no compatible wheel.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the environment from the lockfile:

```bash
uv sync --locked
```

## Run the application

Copy camera exports into `data/<location>/`, then start the main interface:

```bash
uv run wildl_id.py
```

The menu can organise images, detect and classify wildlife, generate visualisations, or run the full pipeline. Original camera photographs are not included.

See the [visualisations](visualisations.md) page for example analyses.

[Home](index.md)
