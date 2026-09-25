from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

import jacobian.math.polynomials.tropical.operations as tropical_operations
from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.tropical._models import MatrixAddRequest
from jacobian.math.polynomials.tropical._tools import TOOLS
from jacobian.math.polynomials.tropical.operations import tropical_matrix_add
from jacobian.math.polynomials.tropical.values import (
    TropicalMatrix,
    TropicalScalar,
    TropicalSemiring,
)


def _matrix(
    convention: str,
    values: tuple[tuple[Fraction | None, ...], ...],
    *,
    base: str = "QQ",
    row_axis: tuple[str, ...] = ("r0", "r1"),
    column_axis: tuple[str, ...] = ("c0", "c1", "c2"),
) -> TropicalMatrix:
    semiring = TropicalSemiring(convention=convention, base=base)  # type: ignore[arg-type]
    infinity = "POSITIVE_INFINITY" if convention == "MIN_PLUS" else "NEGATIVE_INFINITY"
    return TropicalMatrix(
        semiring=semiring,
        row_axis=row_axis,
        column_axis=column_axis,
        entries=tuple(
            tuple(
                TropicalScalar(
                    semiring=semiring,
                    kind="FINITE" if value is not None else infinity,
                    value=(
                        CanonicalRational.from_fraction(value)
                        if value is not None
                        else None
                    ),
                )
                for value in row
            )
            for row in values
        ),
    )


def _oracle_entry(
    convention: str, left: Fraction | None, right: Fraction | None
) -> Fraction | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right) if convention == "MIN_PLUS" else max(left, right)


@pytest.mark.parametrize("convention", ["MIN_PLUS", "MAX_PLUS"])
def test_matrix_add_matches_independent_entrywise_oracle(convention: str) -> None:
    left_values = (
        (Fraction(1, 2), None, Fraction(-3, 2)),
        (Fraction(4), Fraction(7, 3), Fraction(9, 2)),
    )
    right_values = (
        (Fraction(1, 2), Fraction(8), Fraction(-1)),
        (Fraction(2), Fraction(7, 3), None),
    )
    left = _matrix(convention, left_values)
    right = _matrix(convention, right_values)

    actual = tropical_matrix_add(left, right)

    expected = tuple(
        tuple(
            _oracle_entry(convention, x, y)
            for x, y in zip(left_row, right_row, strict=True)
        )
        for left_row, right_row in zip(left_values, right_values, strict=True)
    )
    observed = tuple(
        tuple(
            entry.value.as_fraction() if entry.kind == "FINITE" else None
            for entry in row
        )
        for row in actual.entries
    )
    assert actual.semiring == left.semiring
    assert actual.row_axis == left.row_axis
    assert actual.column_axis == left.column_axis
    assert observed == expected


def test_matrix_add_manifest_result_round_trips_unchanged() -> None:
    left = _matrix(
        "MIN_PLUS",
        ((Fraction(0), Fraction(5), Fraction(2)), (Fraction(4), Fraction(1), None)),
    )
    right = _matrix(
        "MIN_PLUS",
        ((Fraction(3), Fraction(2), Fraction(2)), (Fraction(1), Fraction(1), None)),
    )
    tool = next(
        tool for tool in TOOLS if tool.operation_id == "tropical.matrix.add.compute"
    )

    response = tool.run(MatrixAddRequest(left=left, right=right))

    restored = type(response).model_validate_json(response.model_dump_json())
    assert restored.result == tropical_matrix_add(left, right)


def test_matrix_add_admits_each_input_scalar_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = _matrix(
        "MIN_PLUS",
        ((Fraction(0), Fraction(5), Fraction(2)), (Fraction(4), Fraction(1), None)),
    )
    right = _matrix(
        "MIN_PLUS",
        ((Fraction(3), Fraction(2), Fraction(2)), (Fraction(1), Fraction(1), None)),
    )
    original_admit_scalar = tropical_operations._admit_scalar
    admitted_scalars = 0

    def count_admission(scalar: TropicalScalar, semiring: TropicalSemiring) -> None:
        nonlocal admitted_scalars
        admitted_scalars += 1
        original_admit_scalar(scalar, semiring)

    monkeypatch.setattr(tropical_operations, "_admit_scalar", count_admission)

    tropical_matrix_add(left, right)

    assert admitted_scalars == 2 * 2 * 3


@pytest.mark.parametrize(
    ("rows", "columns", "left_entries", "right_entries"),
    [
        ((), (), (), ()),
        ((), ("c0", "c1"), (), ()),
        (("r0", "r1"), (), ((), ()), ((), ())),
    ],
)
def test_matrix_add_preserves_empty_axes(
    rows: tuple[str, ...],
    columns: tuple[str, ...],
    left_entries: tuple[tuple[Fraction | None, ...], ...],
    right_entries: tuple[tuple[Fraction | None, ...], ...],
) -> None:
    left = _matrix("MIN_PLUS", left_entries, row_axis=rows, column_axis=columns)
    right = _matrix("MIN_PLUS", right_entries, row_axis=rows, column_axis=columns)

    result = tropical_matrix_add(left, right)

    assert result.row_axis == rows
    assert result.column_axis == columns
    assert result.entries == left.entries


def test_matrix_add_rejects_mismatched_semiring_or_labelled_axes() -> None:
    left = _matrix("MIN_PLUS", ((Fraction(0),),), row_axis=("r",), column_axis=("c",))
    other_convention = _matrix(
        "MAX_PLUS", ((Fraction(0),),), row_axis=("r",), column_axis=("c",)
    )
    other_base = _matrix(
        "MIN_PLUS",
        ((Fraction(0),),),
        base="ZZ",
        row_axis=("r",),
        column_axis=("c",),
    )
    other_row = _matrix(
        "MIN_PLUS", ((Fraction(0),),), row_axis=("other",), column_axis=("c",)
    )
    other_column = _matrix(
        "MIN_PLUS", ((Fraction(0),),), row_axis=("r",), column_axis=("other",)
    )

    for right in (other_convention, other_base, other_row, other_column):
        with pytest.raises(OperationDomainValidationError):
            tropical_matrix_add(left, right)
        with pytest.raises(ValidationError):
            MatrixAddRequest(left=left, right=right)
