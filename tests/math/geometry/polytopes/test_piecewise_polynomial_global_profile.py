import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes._models import (
    GlobalPolynomialProfileRequest,
    PieceAssignment,
    PiecewisePolynomialResult,
)
from jacobian.math.geometry.polytopes.complexes._spline import (
    piecewise_polynomial_from_maximal_pieces,
    piecewise_polynomial_global_profile,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    polytopal_complex_closure,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _interval(left: int, right: int, prefix: str) -> RationalVPolytope:
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=("x",)),
        vertices=tuple(
            RationalPolytopeVertex(
                vertex_id=f"{prefix}{index}",
                coordinates=({"num": value, "den": 1},),
            )
            for index, value in enumerate((left, right))
        ),
    )


def _poly(
    coefficients: dict[int, int], variables: tuple[str, ...] = ("x",)
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=coefficient, den=1),
                    exponents=(degree, *(0 for _ in variables[1:])),
                )
                for degree, coefficient in sorted(coefficients.items(), reverse=True)
                if coefficient
            )
        ),
    )


def _function(complex_value, by_cell):
    pieces = tuple(
        PieceAssignment(cell_id=cell.cell_id, polynomial=by_cell[cell.cell_id])
        for cell in complex_value.maximal_cells
    )
    return piecewise_polynomial_from_maximal_pieces(complex_value, pieces)


def test_one_polynomial_on_each_cell_returns_source_bound_global_polynomial():
    complex_value = polytopal_complex_closure(
        (_interval(0, 1, "a"), _interval(1, 2, "b"))
    )
    polynomial = _poly({2: 3, 1: -2, 0: 1})
    function = _function(
        complex_value,
        {cell.cell_id: polynomial for cell in complex_value.maximal_cells},
    )

    result = piecewise_polynomial_global_profile(
        GlobalPolynomialProfileRequest(function=function)
    )

    assert result.status == "GLOBAL_POLYNOMIAL"
    assert result.function == function
    assert result.polynomial == polynomial


def test_continuous_but_different_cell_polynomials_are_not_global():
    complex_value = polytopal_complex_closure(
        (_interval(0, 1, "a"), _interval(1, 2, "b"))
    )
    function = _function(
        complex_value,
        {
            "M0": _poly({1: 1}),
            "M1": _poly({1: 2, 0: -1}),
        },
    )
    assert function.status == "COMPATIBLE"

    result = piecewise_polynomial_global_profile(
        GlobalPolynomialProfileRequest(function=function)
    )

    assert result.status == "NOT_GLOBAL_POLYNOMIAL"
    assert result.polynomial is None


def test_rebuilds_forged_compatibility_and_public_tool_roundtrips():
    complex_value = polytopal_complex_closure(
        (_interval(0, 1, "a"), _interval(1, 2, "b"))
    )
    pieces = (
        PieceAssignment(cell_id="M0", polynomial=_poly({0: 1})),
        PieceAssignment(cell_id="M1", polynomial=_poly({0: 2})),
    )
    forged = PiecewisePolynomialResult(
        complex=complex_value,
        pieces=pieces,
        compatibility=(),
        status="COMPATIBLE",
    )
    with pytest.raises(OperationDomainValidationError, match="not continuous"):
        piecewise_polynomial_global_profile(
            GlobalPolynomialProfileRequest(function=forged)
        )

    single = polytopal_complex_closure((_interval(0, 1, "s"),))
    function = _function(single, {single.maximal_cells[0].cell_id: _poly({0: 1})})
    result = piecewise_polynomial_global_profile(
        GlobalPolynomialProfileRequest(function=function)
    )
    assert result.status == "GLOBAL_POLYNOMIAL"
    assert result.polynomial == _poly({0: 1})
