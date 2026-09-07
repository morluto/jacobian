"""QQ[t] carrier ownership, scalar reuse and empty-axis serialization."""

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.symbolic import RationalPolynomialMatrix
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(variable: str) -> RationalPolynomial:
    return RationalPolynomial(
        variables=(variable,),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=2), exponents=(2,)
                ),
            )
        ),
    )


@pytest.mark.parametrize("rows,columns", [(0, 0), (0, 7), (4, 0)])
def test_empty_axes_retain_the_polynomial_ring(rows: int, columns: int) -> None:
    matrix = RationalPolynomialMatrix(
        variables=("t",),
        row_count=rows,
        column_count=columns,
        entries=tuple(() for _ in range(rows)),
    )
    assert (
        RationalPolynomialMatrix.model_validate_json(matrix.model_dump_json()) == matrix
    )


def test_entries_reuse_the_canonical_polynomial_value() -> None:
    entry = _polynomial("z")
    matrix = RationalPolynomialMatrix(
        variables=("z",),
        row_count=1,
        column_count=1,
        entries=((entry,),),
    )
    restored = RationalPolynomialMatrix.model_validate_json(matrix.model_dump_json())
    assert restored.entries[0][0] == entry
    assert restored.entries[0][0].model_dump() == entry.model_dump()


def test_ring_changes_and_ragged_rows_are_not_implicit_coercions() -> None:
    with pytest.raises(ValidationError, match="ring"):
        RationalPolynomialMatrix(
            variables=("t",),
            row_count=1,
            column_count=1,
            entries=((_polynomial("z"),),),
        )
    with pytest.raises(ValidationError, match="axes"):
        RationalPolynomialMatrix(
            variables=("t",),
            row_count=1,
            column_count=2,
            entries=((_polynomial("t"),),),
        )
    with pytest.raises(ValidationError):
        RationalPolynomialMatrix(
            variables=("t", "z"),
            row_count=0,
            column_count=0,
            entries=(),
        )
