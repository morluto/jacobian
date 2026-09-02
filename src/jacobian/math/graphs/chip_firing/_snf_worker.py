"""One-shot FLINT Smith-normal-form worker for chip-firing critical groups."""

from __future__ import annotations

import hashlib
import sys
from typing import Any

from jacobian.canonical import (
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)


def _diagonal_snf(matrix: list[list[int]]) -> list[int]:
    from flint import fmpz_mat

    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0
    if rows == 0 or cols == 0:
        return []
    diagonal = fmpz_mat(matrix).snf()
    values: list[int] = []
    for index in range(min(rows, cols)):
        value = int(diagonal[index, index])
        values.append(-value if value < 0 else value)
    return values


def main() -> int:
    input_bytes = sys.stdin.buffer.read()
    payload: dict[str, Any] = loads_strict_json(input_bytes)
    if set(payload) != {"matrix"} or not isinstance(payload["matrix"], list):
        raise ValueError("worker request has invalid fields")
    matrix = [
        [parse_canonical_integer(value) for value in row] for row in payload["matrix"]
    ]
    response = {
        "request_digest": hashlib.sha256(input_bytes).hexdigest(),
        "diagonal": [
            format_canonical_integer(value) for value in _diagonal_snf(matrix)
        ],
    }
    sys.stdout.buffer.write(encode_strict_json(response, limits=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
