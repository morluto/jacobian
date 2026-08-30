"""One-shot SymPy Smith-decomposition worker for integral homology."""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any


def main() -> int:
    input_bytes = sys.stdin.buffer.read()
    payload: dict[str, Any] = json.loads(input_bytes)
    source = payload["matrix"]
    rows = payload["row_count"]
    columns = payload["column_count"]

    from jacobian.math.matrices.certified_snf.operations import (
        inverse_unimodular,
        smith_reduce,
    )

    reduction = smith_reduce(source, row_count=rows, column_count=columns)
    response = {
        "request_digest": hashlib.sha256(input_bytes).hexdigest(),
        "diagonal": reduction.diagonal,
        "left": reduction.left,
        "right": reduction.right,
        "rank": reduction.rank,
        "invariant_factors": reduction.invariant_factors,
        "left_determinant": reduction.left_determinant,
        "right_determinant": reduction.right_determinant,
        "left_inverse": inverse_unimodular(reduction.left),
        "right_inverse": inverse_unimodular(reduction.right),
    }
    json.dump(response, sys.stdout, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
