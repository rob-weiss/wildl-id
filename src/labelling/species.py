"""Relevant species for the landing page, forecasts, and daily summaries."""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "relevant_species.json"
RECENT_SPECIES = frozenset(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
