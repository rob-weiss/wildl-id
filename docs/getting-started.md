# Getting Started

Wildlife Identification and Analysis provides Python tools for processing camera-trap images and analysing labelled detections.

## Requirements

Python 3.14 is the only supported version, for the tools, web container and CI. Python 3.15 is not yet supported because PyTorch has no compatible wheel.

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

For manual workflows, reviewer controls, configuration, and other details, see the
[full project guide](project-guide.md).

[Home](index.md)
