"""Standalone FLINT/SymPy worker for admitted unit-disk root counts."""

from __future__ import annotations

import json
import sys
from itertools import pairwise
from typing import Any


def _signature(matrix: list[list[int]]) -> int:
    from sympy.polys.domains import ZZ
    from sympy.polys.matrices.ddm import DDM

    order = len(matrix)
    if not order:
        return 0
    characteristic = DDM(
        [[ZZ(value) for value in row] for row in matrix], (order, order), ZZ
    ).charpoly()
    positive = [1 if value > 0 else -1 for value in characteristic if value]
    negative = [
        (1 if value > 0 else -1) * (-1 if index % 2 else 1)
        for index, value in enumerate(characteristic)
        if value
    ]
    return sum(a != b for a, b in pairwise(positive)) - sum(
        a != b for a, b in pairwise(negative)
    )


def _schur_signature(coefficients: list[int]) -> int:
    n = len(coefficients) - 1
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = sum(
                coefficients[n - i + k] * coefficients[n - j + k]
                - coefficients[i - k] * coefficients[j - k]
                for k in range(min(i, j) + 1)
            )
    return _signature(matrix)


def _real_root_count(coefficients: list[int]) -> int:
    n = len(coefficients) - 1
    derivative = [(i + 1) * coefficients[i + 1] for i in range(n)] + [0]
    matrix = [[0] * n for _ in range(n)]
    for a in range(1, n + 1):
        for b in range(a):
            coefficient = (
                coefficients[a] * derivative[b] - derivative[a] * coefficients[b]
            )
            for k in range(a - b):
                matrix[a - 1 - k][b + k] += coefficient
    return _signature(matrix)


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    from flint import fmpz_poly

    if set(payload) != {"coefficients"} or not isinstance(
        payload["coefficients"], list
    ):
        raise ValueError("malformed unit-disk request")
    coefficients = payload["coefficients"]
    if not coefficients or any(type(value) is not int for value in coefficients):
        raise ValueError("malformed unit-disk request")
    source = fmpz_poly(coefficients)
    inside, boundary, outside = 0, 0, 0
    _, factors = source.factor_squarefree()
    for factor, multiplicity in factors:
        a = [int(value) for value in factor.coeffs()]
        n = len(a) - 1
        difference = _schur_signature(a)
        plus, minus = fmpz_poly([1, 1]), fmpz_poly([1, -1])
        transformed = fmpz_poly([a[-1]])
        power = fmpz_poly([1])
        for coefficient in reversed(a[:-1]):
            power *= minus
            transformed = transformed * plus + coefficient * power
        q = [int(value) for value in transformed.coeffs()]
        real = fmpz_poly(
            [value * (-1) ** (i // 2) if i % 2 == 0 else 0 for i, value in enumerate(q)]
        )
        imaginary = fmpz_poly(
            [value * (-1) ** (i // 2) if i % 2 else 0 for i, value in enumerate(q)]
        )
        common = real.gcd(imaginary)
        on = (
            n
            - int(transformed.degree())
            + _real_root_count([int(value) for value in common.coeffs()])
        )
        inside += multiplicity * ((n - on + difference) // 2)
        boundary += multiplicity * on
        outside += multiplicity * ((n - on - difference) // 2)
    return {
        "status": "ok",
        "inside": inside,
        "on": boundary,
        "outside": outside,
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("malformed unit-disk request")
        response = _run(payload)
    except Exception:
        return 1
    sys.stdout.buffer.write(
        json.dumps(response, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
