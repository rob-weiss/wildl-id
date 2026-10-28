"""Shared metadata-independent image pixel hashing."""

import csv
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image

HASH_VERSION = "rgba-sha256-v1"


def pixel_digest(image):
    pixels = (
        image
        if image.mode.startswith("I") or image.mode == "F"
        else image.convert("RGBA")
    )
    digest = hashlib.sha256(f"{pixels.mode}:{pixels.size}".encode("ascii"))
    digest.update(pixels.tobytes())
    return digest.digest()


def pixel_digest_bytes(content):
    with Image.open(BytesIO(content)) as image:
        return pixel_digest(image)


def pixel_digest_path(path):
    with Image.open(Path(path)) as image:
        return pixel_digest(image)


def read_pixel_hash_cache(path):
    entries = {}
    try:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    if row["hash_version"] != HASH_VERSION:
                        continue
                    digest = bytes.fromhex(row["sha256"])
                    size, mtime_ns = int(row["size"]), int(row["mtime_ns"])
                    if len(digest) != 32 or size < 0:
                        continue
                    entries[row["image_path"]] = (size, mtime_ns, digest)
                except (KeyError, TypeError, ValueError):
                    continue
    except (OSError, UnicodeError, csv.Error):
        return {}
    return entries
