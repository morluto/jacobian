"""Small artifact and digest builders shared across semantic test lanes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jacobian.canonical import canonicalize_json


def artifact_uri(character: str) -> str:
    return "artifact://sha256/" + character * 64


def canonical_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonicalize_json(value)).hexdigest()


def json_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
