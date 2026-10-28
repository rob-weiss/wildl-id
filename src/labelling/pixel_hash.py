"""Shared metadata-independent image pixel hashing."""

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
