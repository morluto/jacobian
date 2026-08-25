"""Dispatch boundaries for exact matrix inertia analysis."""

from __future__ import annotations

import json

import pytest

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.canonical import canonicalize_json, encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.matrices.values import MAX_MATRIX_DIMENSION


def _encoded_inertia_payload_near_limit(offset: int) -> bytes:
    """Encode an inertia request whose normalized dense source echo lands
    exactly ``offset`` bytes below the canonical output limit, so the echo
    plus the reserved envelope may exceed the identical output limit while
    the payload still fits the input limit."""

    import functools

    from jacobian.canonical import CanonicalLimits

    @functools.cache
    def build(offset: int) -> bytes:
        limits = CanonicalLimits()
        dimension = MAX_MATRIX_DIMENSION
        cells = [(r, c) for r in range(dimension) for c in range(r, dimension)]

        def dense_echo(digits: dict[tuple[int, int], int]) -> bytes:
            rows = [
                [
                    {
                        "num": "9" * digits[(min(r, c), max(r, c))],
                        "den": "1",
                    }
                    for c in range(dimension)
                ]
                for r in range(dimension)
            ]
            return encode_strict_json({"domain": "QQ", "entries": rows})

        target = limits.max_output_bytes - offset
        low = len(dense_echo(dict.fromkeys(cells, 1)))
        uniform = max(1, (target - low) // (dimension * dimension))
        digits = dict.fromkeys(cells, uniform)
        gap = target - len(dense_echo(digits))
        first, second = cells[0], cells[1]
        adjusted = digits[first] + gap
        if adjusted < 1:
            digits[second] += adjusted - 1
            adjusted = 1
        elif adjusted > MAX_CANONICAL_RATIONAL_DIGITS:
            digits[second] += adjusted - MAX_CANONICAL_RATIONAL_DIGITS
            adjusted = MAX_CANONICAL_RATIONAL_DIGITS
        assert 1 <= digits[second] <= MAX_CANONICAL_RATIONAL_DIGITS
        digits[first] = adjusted
        assert len(dense_echo(digits)) == target
        encoded = encode_strict_json(
            {
                "dimension": dimension,
                "entries": [
                    {
                        "row": r,
                        "col": c,
                        "value": {"num": "9" * digits[(r, c)], "den": "1"},
                    }
                    for (r, c) in cells
                ],
            }
        )
        assert len(encoded) <= limits.max_input_bytes
        return encoded

    return build(offset)


def test_dispatch_rejects_unfittable_inertia_request_as_typed_error() -> None:
    with pytest.raises(OperationRequestValidationError) as excinfo:
        invoke_operation(
            "matrix.inertia.compute",
            json.loads(_encoded_inertia_payload_near_limit(offset=512)),
            Catalog.open(),
        )
    assert "canonical output limit" in str(excinfo.value.cause)


def test_dispatch_admits_order_33_diagonal_inertia_request() -> None:
    # Order 33 exceeds the shared computation dimension but stays inside the
    # canonical dense rational-matrix order envelope, so a small-entry source
    # there must be admitted end to end with a typed source-bound result.
    payload = {
        "dimension": 33,
        "entries": [
            {"row": r, "col": r, "value": {"num": "1", "den": "1"}} for r in range(33)
        ],
    }
    result = invoke_operation("matrix.inertia.compute", payload, Catalog.open())

    assert result.output["n_positive"] == 33
    assert result.output["n_negative"] == 0
    assert result.output["n_zero"] == 0
    assert result.output["definiteness"] == "positive_definite"
    assert len(result.output["matrix"]["entries"]) == 33


def test_large_fitting_inertia_request_returns_typed_result() -> None:
    digits = "9" * 4096
    payload = {
        "dimension": MAX_MATRIX_DIMENSION,
        "entries": [
            {"row": r, "col": r, "value": {"num": digits, "den": "1"}}
            for r in range(MAX_MATRIX_DIMENSION)
        ],
    }
    assert len(canonicalize_json(payload)) > 100_000
    result = invoke_operation("matrix.inertia.compute", payload, Catalog.open())
    assert result.output["n_positive"] == MAX_MATRIX_DIMENSION
    assert result.output["n_negative"] == 0
    assert result.output["n_zero"] == 0
    assert result.output["definiteness"] == "positive_definite"
    matrix = result.output["matrix"]
    assert matrix["domain"] == "QQ"
    assert len(matrix["entries"]) == MAX_MATRIX_DIMENSION
