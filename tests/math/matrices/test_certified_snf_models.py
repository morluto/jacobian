from __future__ import annotations

from typing import TypedDict

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
    SmithNormalFormCertificate,
)
from jacobian.math.matrices.values import IntegerMatrix


class MatrixWire(TypedDict):
    row_count: int
    column_count: int
    entries: list[list[int]]


class CertificateFields(TypedDict):
    source: IntegerMatrix
    diagonal: IntegerMatrix
    left_transformation: IntegerMatrix
    right_transformation: IntegerMatrix
    rank: int
    invariant_factors: tuple[int, ...]
    left_determinant: int
    right_determinant: int


def _matrix(entries: list[list[int]]) -> MatrixWire:
    return {
        "row_count": len(entries),
        "column_count": len(entries[0]),
        "entries": entries,
    }


def test_certified_smith_request_accepts_a_bounded_integer_rectangle() -> None:
    request = CertifiedSmithNormalFormRequest.model_validate(
        {"matrix": _matrix([[2, 4, 6], [8, 10, 12]])}
    )

    assert request.matrix.row_count == 2
    assert request.matrix.column_count == 3


def test_certified_smith_request_rejects_large_input_scalars() -> None:
    request = CertifiedSmithNormalFormRequest.model_validate(
        {"matrix": _matrix([[(10**33 - 1) // 9]])}
    )

    with pytest.raises(OperationDomainValidationError):
        smith_normal_form_certificate(request.matrix)


def test_certified_smith_request_schema_publishes_the_enforced_dimension_cap() -> None:
    schema = CertifiedSmithNormalFormRequest.model_json_schema()
    matrix_schema = schema["properties"]["matrix"]

    for axis in ("row_count", "column_count"):
        assert matrix_schema["properties"][axis]["maximum"] == 16
        assert matrix_schema["properties"][axis]["minimum"] == 1
        assert matrix_schema["properties"][axis]["type"] == "integer"


def test_certified_smith_result_source_composes_into_a_new_request() -> None:
    source = IntegerMatrix.model_validate(_matrix([[2]]))
    result = CertifiedSmithNormalFormResult._from_kernel(
        certificate=smith_normal_form_certificate(source),
    )

    request = CertifiedSmithNormalFormRequest(matrix=result.certificate.source)

    assert request.matrix is result.certificate.source


def test_certificate_contract_requires_a_canonical_divisibility_diagonal() -> None:
    source = IntegerMatrix.model_validate(_matrix([[2, 0], [0, 6]]))
    identity = IntegerMatrix.model_validate(_matrix([[1, 0], [0, 1]]))

    with pytest.raises(ValidationError):
        SmithNormalFormCertificate(
            source=source,
            diagonal=source,
            left_transformation=identity,
            right_transformation=identity,
            rank=2,
            invariant_factors=(2, 3),
            left_determinant=1,
            right_determinant=1,
        )


def test_zero_dimensional_matrices_remain_explicit_for_chain_boundaries() -> None:
    matrix = IntegerMatrix(
        row_count=0,
        column_count=3,
        entries=(),
    )

    assert matrix.entries == ()
    assert matrix.column_count == 3


def _certified_matrix(entries: list[list[int]]) -> IntegerMatrix:
    return IntegerMatrix(entries=tuple(tuple(row) for row in entries))


def _certificate_kwargs(
    *,
    source: IntegerMatrix | None = None,
    diagonal: IntegerMatrix | None = None,
    left_transformation: IntegerMatrix | None = None,
    right_transformation: IntegerMatrix | None = None,
    rank: int | None = None,
    invariant_factors: tuple[int, ...] | None = None,
    left_determinant: int | None = None,
    right_determinant: int | None = None,
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
            invariant_factors if invariant_factors is not None else (2,)
        ),
        "left_determinant": (left_determinant if left_determinant is not None else 1),
        "right_determinant": (
            right_determinant if right_determinant is not None else 1
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
            invariant_factors=(1, 1),
            left_determinant=-1,
        )
    )

    assert certificate.left_determinant == -1
    assert verify_smith_normal_form_certificate(certificate)
    assert not verify_smith_normal_form_certificate(
        SmithNormalFormCertificate(
            **_certificate_kwargs(
                source=permutation,
                diagonal=identity,
                left_transformation=permutation,
                right_transformation=identity,
                rank=2,
                invariant_factors=(1, 1),
                left_determinant=1,
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
        ("left_determinant", -1),
        ("right_determinant", -1),
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
                invariant_factors=(2, 2),
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
    def square_identity(size: int) -> IntegerMatrix:
        return IntegerMatrix.model_validate(
            _matrix(
                [
                    [1 if row == column else 0 for column in range(size)]
                    for row in range(size)
                ]
            )
        )

    empty_source = IntegerMatrix(
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
            else IntegerMatrix(row_count=0, column_count=0)
        ),
        right_transformation=(
            square_identity(columns)
            if columns
            else IntegerMatrix(row_count=0, column_count=0)
        ),
        rank=0,
        invariant_factors=(),
    )

    certificate = SmithNormalFormCertificate(**kwargs)

    assert certificate.rank == 0
    assert verify_smith_normal_form_certificate(certificate)
    assert not verify_smith_normal_form_certificate(
        SmithNormalFormCertificate.model_validate({**kwargs, "left_determinant": -1})
    )


def test_certificate_contract_replays_rank_zero_determinants_without_formatting_them() -> (
    None
):
    tall_source = IntegerMatrix(row_count=2, column_count=0, entries=((), ()))
    empty_right = IntegerMatrix(row_count=0, column_count=0)
    permutation = _certified_matrix([[0, 1], [1, 0]])
    inflated = _certified_matrix([[(10**64 - 1) // 9, 0], [0, 1]])
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
            "left_determinant": -1,
        }
    )

    assert certificate.left_determinant == -1
    assert verify_smith_normal_form_certificate(certificate)
    assert not verify_smith_normal_form_certificate(
        SmithNormalFormCertificate.model_validate(
            {
                **kwargs,
                "left_transformation": inflated,
                "left_determinant": -1,
            }
        )
    )


@pytest.mark.parametrize(("rows", "columns"), [(0, 3), (3, 0), (0, 0)])
def test_plain_smith_preserves_canonical_empty_shape(rows: int, columns: int) -> None:
    from jacobian.math.matrices.operations import smith_normal_form_result

    source = IntegerMatrix(row_count=rows, column_count=columns, entries=((),) * rows)
    source = IntegerMatrix.model_validate_json(source.model_dump_json())
    result = smith_normal_form_result(source)
    assert result.normal_form == source
    assert result.rank == 0
    assert result.invariant_factors == ()
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_inferred_integer_dimensions_do_not_advertise_literal_defaults() -> None:
    from jsonschema import Draft202012Validator

    schema = IntegerMatrix.model_json_schema()
    for name in ("row_count", "column_count"):
        assert "default" not in schema["properties"][name]
    payload = {"entries": [["2", "3"]]}
    Draft202012Validator(schema).validate(payload)
    matrix = IntegerMatrix.model_validate_json(__import__("json").dumps(payload))
    assert (matrix.row_count, matrix.column_count) == (1, 2)
