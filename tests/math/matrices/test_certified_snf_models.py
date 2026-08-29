from __future__ import annotations

from typing import Literal, TypedDict

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.certified_snf._models import (
    CertifiedSmithNormalFormRequest,
    CertifiedSmithNormalFormResult,
)
from jacobian.math.matrices.certified_snf.operations import (
    smith_normal_form_certificate,
    verify_smith_normal_form_certificate,
)
from jacobian.math.matrices.certified_snf.values import (
    CertifiedIntegerMatrix,
    SmithNormalFormCertificate,
)


class MatrixWire(TypedDict):
    row_count: int
    column_count: int
    entries: list[list[str]]


class CertificateFields(TypedDict):
    source: CertifiedIntegerMatrix
    diagonal: CertifiedIntegerMatrix
    left_transformation: CertifiedIntegerMatrix
    right_transformation: CertifiedIntegerMatrix
    rank: int
    invariant_factors: tuple[str, ...]
    left_determinant: Literal["-1", "1"]
    right_determinant: Literal["-1", "1"]


def _matrix(entries: list[list[int | str]]) -> MatrixWire:
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
    request = CertifiedSmithNormalFormRequest.model_validate(
        {"matrix": _matrix([["1" * 33]])}
    )

    with pytest.raises(OperationDomainValidationError):
        smith_normal_form_certificate(request.matrix)


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

    with pytest.raises(ValidationError):
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


def _certified_matrix(entries: list[list[int | str]]) -> CertifiedIntegerMatrix:
    return CertifiedIntegerMatrix.model_validate(_matrix(entries))


def _certificate_kwargs(
    *,
    source: CertifiedIntegerMatrix | None = None,
    diagonal: CertifiedIntegerMatrix | None = None,
    left_transformation: CertifiedIntegerMatrix | None = None,
    right_transformation: CertifiedIntegerMatrix | None = None,
    rank: int | None = None,
    invariant_factors: tuple[str, ...] | None = None,
    left_determinant: Literal["-1", "1"] | None = None,
    right_determinant: Literal["-1", "1"] | None = None,
) -> CertificateFields:
    return {
        "source": source if source is not None else _certified_matrix([[2]]),
        "diagonal": (diagonal if diagonal is not None else _certified_matrix([[2]])),
        "left_transformation": (
            left_transformation
            if left_transformation is not None
            else _certified_matrix([[1]])
        ),
        "right_transformation": (
            right_transformation
            if right_transformation is not None
            else _certified_matrix([[1]])
        ),
        "rank": rank if rank is not None else 1,
        "invariant_factors": (
            invariant_factors if invariant_factors is not None else ("2",)
        ),
        "left_determinant": (left_determinant if left_determinant is not None else "1"),
        "right_determinant": (
            right_determinant if right_determinant is not None else "1"
        ),
    }


def test_certificate_verifier_rejects_the_exact_transformation_relation() -> None:
    certificate = SmithNormalFormCertificate(
        **_certificate_kwargs(
            left_transformation=_certified_matrix([[2]]),
        )
    )

    assert not verify_smith_normal_form_certificate(certificate)


def test_certificate_contract_replays_declared_determinant_signs() -> None:
    permutation = _certified_matrix([[0, 1], [1, 0]])
    identity = _certified_matrix([[1, 0], [0, 1]])

    certificate = SmithNormalFormCertificate(
        **_certificate_kwargs(
            source=permutation,
            diagonal=identity,
            left_transformation=permutation,
            right_transformation=identity,
            rank=2,
            invariant_factors=("1", "1"),
            left_determinant="-1",
        )
    )

    assert certificate.left_determinant == "-1"
    assert verify_smith_normal_form_certificate(certificate)
    assert not verify_smith_normal_form_certificate(
        SmithNormalFormCertificate(
            **_certificate_kwargs(
                source=permutation,
                diagonal=identity,
                left_transformation=permutation,
                right_transformation=identity,
                rank=2,
                invariant_factors=("1", "1"),
                left_determinant="1",
            )
        )
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", _certified_matrix([[4]])),
        ("diagonal", _certified_matrix([[4]])),
        ("left_transformation", _certified_matrix([[-1]])),
        ("right_transformation", _certified_matrix([[3]])),
        ("rank", 0),
        ("invariant_factors", ("4",)),
        ("left_determinant", "-1"),
        ("right_determinant", "-1"),
    ],
)
def test_certificate_verifier_fails_closed_on_any_claim_field_mutation(
    field: str,
    value: object,
) -> None:
    SmithNormalFormCertificate(**_certificate_kwargs())

    candidate = {**_certificate_kwargs(), field: value}
    if field in {"diagonal", "rank", "invariant_factors"}:
        with pytest.raises(ValidationError):
            SmithNormalFormCertificate.model_validate(candidate)
    else:
        assert not verify_smith_normal_form_certificate(
            SmithNormalFormCertificate.model_validate(candidate)
        )


def test_certificate_contract_replays_rectangular_sources() -> None:
    certificate = SmithNormalFormCertificate(
        **_certificate_kwargs(
            source=_certified_matrix([[2, 6]]),
            diagonal=_certified_matrix([[2, 0]]),
            right_transformation=_certified_matrix([[1, -3], [0, 1]]),
        )
    )

    assert certificate.rank == 1
    assert verify_smith_normal_form_certificate(certificate)
    assert not verify_smith_normal_form_certificate(
        SmithNormalFormCertificate(
            **_certificate_kwargs(
                source=_certified_matrix([[2, 6]]),
                diagonal=_certified_matrix([[2, 0]]),
                right_transformation=_certified_matrix([[1, -6], [0, 1]]),
            )
        )
    )


def test_certificate_contract_replays_rank_deficient_sources() -> None:
    certificate = SmithNormalFormCertificate(
        **_certificate_kwargs(
            source=_certified_matrix([[2, 4], [0, 0]]),
            diagonal=_certified_matrix([[2, 0], [0, 0]]),
            left_transformation=_certified_matrix([[1, 0], [0, 1]]),
            right_transformation=_certified_matrix([[1, -2], [0, 1]]),
        )
    )

    assert certificate.rank == 1
    assert verify_smith_normal_form_certificate(certificate)
    assert not verify_smith_normal_form_certificate(
        SmithNormalFormCertificate(
            **_certificate_kwargs(
                source=_certified_matrix([[2, 4], [0, 0]]),
                diagonal=_certified_matrix([[2, 0], [0, 2]]),
                left_transformation=_certified_matrix([[1, 0], [0, 1]]),
                right_transformation=_certified_matrix([[1, -2], [0, 1]]),
                rank=2,
                invariant_factors=("2", "2"),
            )
        )
    )


@pytest.mark.parametrize(
    ("rows", "columns"),
    [(0, 3), (2, 0)],
)
def test_certificate_contract_replays_zero_dimensional_boundaries(
    rows: int,
    columns: int,
) -> None:
    def square_identity(size: int) -> CertifiedIntegerMatrix:
        return CertifiedIntegerMatrix.model_validate(
            _matrix(
                [
                    [1 if row == column else 0 for column in range(size)]
                    for row in range(size)
                ]
            )
        )

    empty_source = CertifiedIntegerMatrix(
        row_count=rows,
        column_count=columns,
        entries=tuple(() for _ in range(rows)),
    )
    kwargs = _certificate_kwargs(
        source=empty_source,
        diagonal=empty_source,
        left_transformation=(
            square_identity(rows)
            if rows
            else CertifiedIntegerMatrix(row_count=0, column_count=0)
        ),
        right_transformation=(
            square_identity(columns)
            if columns
            else CertifiedIntegerMatrix(row_count=0, column_count=0)
        ),
        rank=0,
        invariant_factors=(),
    )

    certificate = SmithNormalFormCertificate(**kwargs)

    assert certificate.rank == 0
    assert not verify_smith_normal_form_certificate(certificate)
    assert not verify_smith_normal_form_certificate(
        SmithNormalFormCertificate.model_validate({**kwargs, "left_determinant": "-1"})
    )


def test_certificate_contract_replays_rank_zero_determinants_without_formatting_them() -> (
    None
):
    tall_source = CertifiedIntegerMatrix(row_count=2, column_count=0, entries=((), ()))
    empty_right = CertifiedIntegerMatrix(row_count=0, column_count=0)
    permutation = _certified_matrix([[0, 1], [1, 0]])
    inflated = _certified_matrix([["1" * 64, 0], [0, 1]])
    kwargs = _certificate_kwargs(
        source=tall_source,
        diagonal=tall_source,
        right_transformation=empty_right,
        rank=0,
        invariant_factors=(),
    )

    certificate = SmithNormalFormCertificate.model_validate(
        {
            **kwargs,
            "left_transformation": permutation,
            "left_determinant": "-1",
        }
    )

    assert certificate.left_determinant == "-1"
    assert not verify_smith_normal_form_certificate(certificate)
    assert not verify_smith_normal_form_certificate(
        SmithNormalFormCertificate.model_validate(
            {
                **kwargs,
                "left_transformation": inflated,
                "left_determinant": "-1",
            }
        )
    )
