from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.matrices.certified_snf._models import (
    CertifiedSmithNormalFormRequest,
    CertifiedSmithNormalFormResult,
)
from jacobian.math.matrices.certified_snf.values import (
    CertifiedIntegerMatrix,
    SmithNormalFormCertificate,
)


def _matrix(entries: list[list[int | str]]) -> dict[str, object]:
    return {
        "row_count": len(entries),
        "column_count": len(entries[0]),
        "entries": [[str(value) for value in row] for row in entries],
    }


def test_certified_smith_request_accepts_a_bounded_integer_rectangle() -> None:
    request = CertifiedSmithNormalFormRequest.model_validate(
        {"matrix": _matrix([[2, 4, 6], [8, 10, 12]])}
    )

    assert request.matrix.row_count == 2
    assert request.matrix.column_count == 3


def test_certified_smith_request_rejects_large_input_scalars() -> None:
    with pytest.raises(ValidationError, match="at most 32 decimal digits"):
        CertifiedSmithNormalFormRequest.model_validate(
            {"matrix": _matrix([["1" * 33]])}
        )


def test_certified_smith_request_schema_publishes_the_enforced_dimension_cap() -> None:
    schema = CertifiedSmithNormalFormRequest.model_json_schema()
    matrix_schema = schema["properties"]["matrix"]

    assert matrix_schema["properties"]["row_count"] == {
        "maximum": 16,
        "minimum": 1,
        "title": "Row Count",
        "type": "integer",
    }
    assert matrix_schema["properties"]["column_count"] == {
        "maximum": 16,
        "minimum": 1,
        "title": "Column Count",
        "type": "integer",
    }


def test_certified_smith_result_source_composes_into_a_new_request() -> None:
    source = CertifiedIntegerMatrix.model_validate(_matrix([[2]]))
    identity = CertifiedIntegerMatrix.model_validate(_matrix([[1]]))
    result = CertifiedSmithNormalFormResult(
        certificate=SmithNormalFormCertificate(
            source=source,
            diagonal=source,
            left_transformation=identity,
            right_transformation=identity,
            rank=1,
            invariant_factors=("2",),
            left_determinant="1",
            right_determinant="1",
        )
    )

    request = CertifiedSmithNormalFormRequest(matrix=result.certificate.source)

    assert request.matrix is result.certificate.source


def test_certificate_contract_requires_a_canonical_divisibility_diagonal() -> None:
    source = CertifiedIntegerMatrix.model_validate(_matrix([[2, 0], [0, 6]]))
    identity = CertifiedIntegerMatrix.model_validate(_matrix([[1, 0], [0, 1]]))

    with pytest.raises(ValidationError, match="positive divisibility diagonal"):
        SmithNormalFormCertificate(
            source=source,
            diagonal=source,
            left_transformation=identity,
            right_transformation=identity,
            rank=2,
            invariant_factors=("2", "3"),
            left_determinant="1",
            right_determinant="1",
        )


def test_zero_dimensional_matrices_remain_explicit_for_chain_boundaries() -> None:
    matrix = CertifiedIntegerMatrix(
        row_count=0,
        column_count=3,
        entries=(),
    )

    assert matrix.entries == ()
    assert matrix.column_count == 3
