"""Source-bound matrix relations survive structural serialization."""

import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.matrices import (
    verify_adjugate,
    verify_inverse,
    verify_kronecker_product,
    verify_nullspace,
    verify_partial_trace,
    verify_product,
)
from jacobian.math.matrices.operations import (
    adjugate_result,
    inverse_result,
    kronecker_product_result,
    nullspace_result,
    partial_trace_result,
    product_result,
    rank_result,
)
from jacobian.math.matrices.values import (
    IntegerMatrix,
    RationalMatrix,
    SparseRationalMatrix,
    SparseRationalMatrixEntry,
)


def _q(value: int) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(value, 1)


@pytest.mark.parametrize(
    "kind", ["product", "inverse", "adjugate", "kronecker", "partial_trace"]
)
def test_matrix_claim_checks_its_retained_source(kind: str) -> None:
    integer = IntegerMatrix(entries=((2,),))
    rational = RationalMatrix(entries=((_q(2),),))
    cases: dict[str, tuple[Any, Callable[..., bool], str]] = {
        "product": (product_result(rational, rational), verify_product, "product"),
        "inverse": (inverse_result(integer), verify_inverse, "inverse"),
        "adjugate": (adjugate_result(integer), verify_adjugate, "adjugate"),
        "kronecker": (
            kronecker_product_result(rational, rational),
            verify_kronecker_product,
            "product",
        ),
        "partial_trace": (
            partial_trace_result(rational, 1, 1),
            verify_partial_trace,
            "reduced_matrix",
        ),
    }
    result, verify, field = cases[kind]
    payload = result.model_dump(mode="json")
    assert verify(type(result).model_validate_json(json.dumps(payload)))
    payload[field]["entries"] = [
        ["7" if kind == "adjugate" else {"num": "7", "den": "1"}]
    ]
    assert not verify(type(result).model_validate_json(json.dumps(payload)))


def test_product_rejects_mismatched_source_axes() -> None:
    matrix = RationalMatrix(entries=((_q(1),),))
    result = product_result(matrix, matrix)
    payload = result.model_dump(mode="json")
    payload["right"] = RationalMatrix(entries=(), column_count=1).model_dump(
        mode="json"
    )
    with pytest.raises(ValidationError, match="inner axes"):
        type(result).model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "source",
    [RationalMatrix(entries=(), column_count=3), RationalMatrix(entries=((_q(1),),))],
)
def test_nullspace_returns_composable_qq_matrix(source: RationalMatrix) -> None:
    result = nullspace_result(source)
    payload = result.model_dump(mode="json")
    assert "basis_vectors" not in payload
    basis = RationalMatrix.model_validate_json(json.dumps(payload["basis_matrix"]))
    assert basis.column_count == source.column_count
    assert rank_result(basis).rank == result.nullity


def test_serialized_nullspace_verifier_rejects_forged_basis() -> None:
    source = RationalMatrix(entries=((_q(1), _q(2)), (_q(2), _q(4))))
    result = nullspace_result(source)
    payload = result.model_dump(mode="json")
    assert verify_nullspace(type(result).model_validate_json(result.model_dump_json()))
    payload["basis_matrix"]["entries"][0][0] = {"num": "1", "den": "1"}
    import json

    assert not verify_nullspace(type(result).model_validate_json(json.dumps(payload)))


def test_sparse_nullspace_retains_largest_supported_column_axis() -> None:
    order = 8192
    source = SparseRationalMatrix(
        row_count=order,
        column_count=order,
        entries=tuple(
            SparseRationalMatrixEntry(row=i, column=i, value=_q(1))
            for i in range(order)
        ),
    )
    result = nullspace_result(source)
    assert result.nullity == 0
    basis = RationalMatrix.model_validate_json(result.basis_matrix.model_dump_json())
    assert basis.row_count == 0 and basis.column_count == order
