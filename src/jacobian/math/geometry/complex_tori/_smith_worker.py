"""Killable FLINT worker for the Riemann-form Smith normal form."""

from __future__ import annotations

import hashlib
import sys

from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
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
    payload = loads_strict_json(
        input_bytes,
        limits=CanonicalLimits(max_input_bytes=len(input_bytes)),
    )
    entries = payload["entries"]
    matrix = tuple(tuple(_decode_integer(value) for value in row) for row in entries)
    normal_form = integer_smith_normal_form(matrix)
    sys.stdout.buffer.write(
        encode_strict_json(
            {
                "request_digest": hashlib.sha256(input_bytes).hexdigest(),
                "normal_form": [
                    [format_canonical_integer(value) for value in row]
                    for row in normal_form
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
