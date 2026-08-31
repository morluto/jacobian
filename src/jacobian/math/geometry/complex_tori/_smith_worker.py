"""Killable FLINT worker for the Riemann-form Smith normal form."""

from __future__ import annotations

import hashlib
import json
import sys

from jacobian.canonical import (
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.math.matrices._flint import integer_smith_normal_form


def _decode_integer(value: object) -> int:
    if not isinstance(value, str):
        raise ValueError("worker integer must be a canonical string")
    parsed = parse_canonical_integer(value)
    if format_canonical_integer(parsed) != value:
        raise ValueError("worker integer is not canonical")
    return parsed


def main() -> None:
    input_bytes = sys.stdin.buffer.read()
    payload = loads_strict_json(input_bytes)
    entries = payload["entries"]
    matrix = tuple(
        tuple(_decode_integer(value) for value in row) for row in entries
    )
    normal_form = integer_smith_normal_form(matrix)
    json.dump(
        {
            "request_digest": hashlib.sha256(input_bytes).hexdigest(),
            "normal_form": [
                [format_canonical_integer(value) for value in row]
                for row in normal_form
            ],
        },
        sys.stdout,
        separators=(",", ":"),
    )


if __name__ == "__main__":
    main()
