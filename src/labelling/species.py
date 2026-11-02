"""Relevant species for the landing page, forecasts, and daily summaries."""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "relevant_species.json"
RECENT_SPECIES = frozenset(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
GAME_SPECIES = frozenset(
    json.loads(CONFIG_PATH.with_name("game_species.json").read_text(encoding="utf-8"))
)
LABEL_EMOJIS = {
    "roe deer": "🦌",
    "red deer": "🦌",
    "wild boar": "🐗",
    "pigeon": "🐦",
    "badger": "🦡",
    "dog": "🐕",
    "fox": "🦊",
    "hare": "🐇",
    "squirrel": "🐿️",
    "raccoon": "🦝",
    "crow": "🐦‍⬛",
    "human": "🧍",
    "none": "🚫",
    "unknown": "❓",
    "cat": "🐈",
    "vehicle": "🚗",
    "bird": "🐦",
    "marten": "🦦",
}


def normalize_species(label):
    label = str(label).strip().lower()
    if label.replace("_", " ") in {
        "song bird",
        "songbird",
        "woodpecker",
        "buzzard",
        "falcon",
        "owl",
        "jay",
    }:
        return "bird"
    return {
        "equid": "horse",
        "lagomorph": "hare",
        "micromammal": "mouse",
        "mustelid": "marten",
    }.get(label, label)


def species_emoji(label):
    return LABEL_EMOJIS.get(normalize_species(label), "🐾")


def pluralize_species(label, count):
    label = str(label).strip()
    if count == 1 or label.casefold().endswith("deer"):
        return label
    last_word = label.rsplit(maxsplit=1)[-1]
    if last_word.endswith(("s", "x", "z", "ch", "sh")):
        return f"{label}es"
    if (
        last_word.endswith("y")
        and len(last_word) > 1
        and last_word[-2].casefold() not in "aeiou"
    ):
        return f"{label[:-1]}ies"
    return f"{label}s"
