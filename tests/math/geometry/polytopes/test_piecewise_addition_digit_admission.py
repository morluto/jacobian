"""Addition refuses aggregate scalar growth without formatting caller integers."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes import _spline as spline_kernel
from jacobian.math.geometry.polytopes.complexes._models import (
    PieceAssignment,
    PiecewisePolynomialAdditionRequest,
    PiecewisePolynomialResult,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    polytopal_complex_closure,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def test_addition_rejects_large_aggregate_without_decimal_formatting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interval = RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x",)),
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=f"v{x}", coordinates=(CanonicalRational(num=x, den=1),)
            )
            for x in (0, 1)
        ),
    )
    complex_ = polytopal_complex_closure((interval,))
    # 2,048 full-width numerators already fill the 64 Mi-digit output budget;
    # their denominator digits make the real aggregate strictly over budget.
    coefficient = CanonicalRational(num=2**108850, den=1)
    terms = tuple(
        RationalPolynomialTerm(coefficient=coefficient, exponents=(exponent,))
        for exponent in range(2047, -1, -1)
    )

    def function(
        entries: tuple[RationalPolynomialTerm, ...],
    ) -> PiecewisePolynomialResult:
        return PiecewisePolynomialResult(
            complex=complex_,
            pieces=(
                PieceAssignment(
                    cell_id=complex_.maximal_cells[0].cell_id,
                    polynomial=RationalPolynomial(
                        variables=("x",),
                        polynomial=SparseRationalPolynomial(terms=entries),
                    ),
                ),
            ),
            compatibility=(),
            status="COMPATIBLE",
        )

    request = PiecewisePolynomialAdditionRequest(
        left=function(terms), right=function(())
    )

    def forbidden_format(value: int) -> int:
        raise AssertionError("addition admission formatted a caller-sized integer")

    # This sentinel observes the allocation boundary; the mathematical input,
    # declared resource limit and domain computation are unchanged.
    monkeypatch.setattr(spline_kernel, "decimal_digit_width", forbidden_format)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        spline_kernel.piecewise_polynomial_add(request)
    assert exc_info.value.errors()[0]["type"] == "polytopal_complex.addition_output"
