"""Killable SymPy worker for one exact algebraic inertia computation."""

from __future__ import annotations

import hashlib
import sys
from typing import Any

from sympy import QQ, CRootOf, Poly, Symbol

from jacobian.canonical import (
    encode_strict_json,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.math._root_isolation import strict_root_count


def _fraction(value: object) -> object:
    if not isinstance(value, str) or value.count("/") != 1:
        raise ValueError("worker coefficient must be a fraction string")
    numerator, denominator = value.split("/")
    return QQ(
        parse_canonical_integer(numerator),
        parse_canonical_integer(denominator),
    )


def _field(payload: dict[str, object]) -> tuple[Poly, Any]:
    presentation = payload["presentation"]
    if not isinstance(presentation, dict):
        raise ValueError("worker presentation must be an object")
    coefficients = presentation["coefficients_descending"]
    root_index = payload["real_root_index"]
    if not isinstance(coefficients, list) or not isinstance(root_index, int):
        raise ValueError("worker field data is malformed")
    x = Symbol("x")
    defining = Poly.from_list(
        [parse_canonical_integer(coefficient) for coefficient in coefficients],
        gens=x,
        domain=QQ,
    )
    return defining, QQ.algebraic_field(CRootOf(defining, root_index), alias="alpha")


def _representative(coefficients: object, *, x: Symbol) -> Poly:
    if not isinstance(coefficients, list):
        raise ValueError("worker element coefficients must be a list")
    return Poly.from_list(
        [_fraction(coefficient) for coefficient in coefficients],
        gens=x,
        domain=QQ,
    )


def _sign(defining: Poly, representative: Poly, root_index: int) -> int:
    if representative.is_zero:
        return 0
    roots_seen = 0
    for (lower, upper), _multiplicity in (defining * representative).intervals():
        if not strict_root_count(defining, lower, upper):
            continue
        if roots_seen != root_index:
            roots_seen += 1
            continue
        sample = lower if lower == upper else (lower + upper) / 2
        exact_value = representative.eval(sample)
        return 1 if exact_value > 0 else -1 if exact_value < 0 else 0
    raise ValueError("selected real embedding was not isolated")


def _inertia(payload: dict[str, object]) -> tuple[int, int, int]:
    from jacobian.math.matrices.analysis.operations import (
        _diagonal_inertia,
        _symmetric_algebraic_inertia,
    )

    defining, field = _field(payload)
    root_index = payload["real_root_index"]
    matrix_payload = payload["matrix"]
    regime = payload["regime"]
    if not isinstance(root_index, int) or not isinstance(matrix_payload, list):
        raise ValueError("worker inertia request is malformed")
    if any(not isinstance(row, list) for row in matrix_payload):
        raise ValueError("worker inertia matrix rows must be lists")
    order = len(matrix_payload)
    if any(len(row) != order for row in matrix_payload):
        raise ValueError("worker inertia matrix must be square")
    x = defining.gens[0]
    matrix = [
        [field.new(_representative(value, x=x).all_coeffs()) for value in row]
        for row in matrix_payload
    ]

    def sign(value: object) -> int:
        coefficients = list(value.to_list())  # type: ignore[attr-defined]
        representative = Poly.from_list(coefficients, gens=x, domain=QQ)
        return _sign(defining, representative, root_index)

    if regime == "DIAGONAL":
        return _diagonal_inertia(
            [matrix[index][index] for index in range(len(matrix))],
            sign=sign,
            checkpoint=lambda _stage: None,
        )
    if regime == "GENERAL":
        return _symmetric_algebraic_inertia(
            matrix,
            sign=sign,
            checkpoint=lambda _stage: None,
        )
    raise ValueError("worker inertia regime is invalid")


def main() -> None:
    input_bytes = sys.stdin.buffer.read()
    payload = loads_strict_json(input_bytes)
    if not isinstance(payload, dict):
        raise ValueError("worker request must be an object")
    n_positive, n_negative, n_zero = _inertia(payload)
    sys.stdout.buffer.write(
        encode_strict_json(
            {
                "request_digest": hashlib.sha256(input_bytes).hexdigest(),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "n_zero": n_zero,
            }
        )
    )


if __name__ == "__main__":
    main()
