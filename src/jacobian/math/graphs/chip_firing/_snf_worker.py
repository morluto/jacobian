"""One-shot exact normal-form worker for chip-firing critical groups."""

from __future__ import annotations

import hashlib
import sys
import time
from typing import Any

from jacobian.canonical import (
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.catalog.models import OperationResourceAdmissionError


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


def _coordinate_snf(
    matrix: list[list[int]], divisor: list[int], deadline: float
) -> tuple[list[int], list[int]]:
    from sympy import ZZ, Matrix
    from sympy.matrices.normalforms import smith_normal_decomp

    from jacobian._execution import (
        bind_request_deadline,
        request_checkpoint,
        request_execution,
    )
    from jacobian.math.graphs.chip_firing._hermite import prepare_smith_coordinates

    with request_execution(started_at=time.monotonic()):
        bind_request_deadline(deadline)
        residual, image, units = prepare_smith_coordinates(matrix, divisor, deadline)
        if not residual:
            return [1] * units, []
        # Admission follows exact reduction and precedes Smith expansion.
        # Only final factors/residues leave the worker, never transforms.
        diagonal, left, _ = smith_normal_decomp(Matrix(residual), domain=ZZ)
        factors = [int(diagonal[i, i]) for i in range(len(residual))]
        transported = left * Matrix(image)
        coordinates = [int(transported[i]) % d for i, d in enumerate(factors) if d > 1]
        request_checkpoint("after Smith coordinate projection")
        return [1] * units + factors, coordinates


def main() -> int:
    input_bytes = sys.stdin.buffer.read()
    payload: dict[str, Any] = loads_strict_json(input_bytes)
    if set(payload) not in (
        {"matrix"},
        {"matrix", "divisor", "deadline"},
    ) or not isinstance(payload["matrix"], list):
        raise ValueError("worker request has invalid fields")
    matrix = [
        [parse_canonical_integer(value) for value in row] for row in payload["matrix"]
    ]
    if "divisor" in payload:
        try:
            factors, coordinates = _coordinate_snf(
                matrix,
                [parse_canonical_integer(v) for v in payload["divisor"]],
                float(payload["deadline"]),
            )
        except OperationResourceAdmissionError as exc:
            sys.stdout.buffer.write(
                encode_strict_json(
                    {
                        "request_digest": hashlib.sha256(input_bytes).hexdigest(),
                        "resource_error": str(exc),
                    }
                )
            )
            return 0
    else:
        factors = _diagonal_snf(matrix)
        coordinates = []
    response = {
        "request_digest": hashlib.sha256(input_bytes).hexdigest(),
        "diagonal": [format_canonical_integer(value) for value in factors],
    }
    if "divisor" in payload:
        response["coordinates"] = [format_canonical_integer(v) for v in coordinates]
    sys.stdout.buffer.write(encode_strict_json(response, limits=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
