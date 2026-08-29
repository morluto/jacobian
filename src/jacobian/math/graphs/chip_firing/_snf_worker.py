"""One-shot FLINT Smith-normal-form worker for chip-firing critical groups."""

from __future__ import annotations

import json
import sys
from typing import Any


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
    payload: dict[str, Any] = json.load(sys.stdin)
    matrix = payload["matrix"]
    response = {"ok": True, "diagonal": _diagonal_snf(matrix)}
    json.dump(response, sys.stdout, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
