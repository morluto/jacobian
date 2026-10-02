"""Reduced spline outputs retain the canonical rational component envelope."""

from fractions import Fraction

import pytest

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry.polytopes._models import RationalVPolytope
from jacobian.math.geometry.polytopes.complexes._models import (
    PieceAssignment,
    PiecewisePolynomialResult,
    PiecewisePolynomialScalarMultiplicationRequest,
    SplineCoordinatesRequest,
    SplineCoordinatesResult,
)
from jacobian.math.geometry.polytopes.complexes._spline import (
    piecewise_polynomial_from_maximal_pieces,
    piecewise_polynomial_scalar_multiply,
    spline_coordinates,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    polytopal_complex_closure,
)
from jacobian.math.polynomials.values import RationalPolynomial


@pytest.fixture(scope="module")
def large_integer() -> int:
    return int(10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1))


def _constant(value: Fraction) -> PiecewisePolynomialResult:
    complex_value = polytopal_complex_closure(
        (
            RationalVPolytope.model_validate(
                {
                    "space": {"axes": ["x"]},
                    "vertices": [
                        {"vertex_id": "a", "coordinates": [{"num": 0, "den": 1}]},
                        {"vertex_id": "b", "coordinates": [{"num": 1, "den": 1}]},
                    ],
                }
            ),
        )
    )
    polynomial = RationalPolynomial.model_validate(
        {
            "variables": ["x"],
            "polynomial": {
                "terms": [
                    {
                        "coefficient": CanonicalRational.from_fraction(value),
                        "exponents": [0],
                    }
                ]
            },
        }
    )
    return piecewise_polynomial_from_maximal_pieces(
        complex_value,
        (
            PieceAssignment(
                cell_id=complex_value.maximal_cells[0].cell_id, polynomial=polynomial
            ),
        ),
    )


@pytest.mark.parametrize("public", [False, True], ids=["native", "dispatch"])
@pytest.mark.parametrize(
    "case",
    [
        "cancel",
        "reverse_cancel",
        "zero",
        "unit",
        "wide_scalar",
        "double",
        "negative_double",
        "half_reciprocal",
        "boundary_product",
    ],
)
def test_scalar_multiplication_admits_reduced_boundary(
    large_integer: int, public: bool, case: str
) -> None:
    n = large_integer
    coefficient, scalar, expected = {
        "double": (Fraction(n), Fraction(2), Fraction(2 * n)),
        "negative_double": (Fraction(-n), Fraction(2), Fraction(-2 * n)),
        "half_reciprocal": (Fraction(1, n), Fraction(1, 2), Fraction(1, 2 * n)),
        "boundary_product": (Fraction(5 * n - 1), Fraction(2), Fraction(10 * n - 2)),
        "cancel": (Fraction(n), Fraction(1, n), Fraction(1)),
        "reverse_cancel": (Fraction(1, n), Fraction(-n), Fraction(-1)),
        "zero": (Fraction(n), Fraction(0), Fraction(0)),
        "unit": (Fraction(-n), Fraction(1), Fraction(-n)),
        "wide_scalar": (Fraction(n + 1, n), Fraction(n, n + 1), Fraction(1)),
    }[case]
    request = PiecewisePolynomialScalarMultiplicationRequest(
        function=_constant(coefficient),
        scalar=CanonicalRational.from_fraction(scalar),
    )
    if public:
        result = PiecewisePolynomialResult.model_validate_json(
            encode_strict_json(
                invoke_operation(
                    "piecewise_polynomial.scalar_multiply.compute",
                    request.model_dump(mode="json"),
                    Catalog.open(),
                ).output
            )
        )
    else:
        result = piecewise_polynomial_scalar_multiply(request)
    revived = PiecewisePolynomialResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json"))
    )
    terms = revived.pieces[0].polynomial.polynomial.terms
    assert (terms[0].coefficient.as_fraction() if terms else Fraction(0)) == expected
    # Consume the serialized polynomial through the newly repaired coordinate path.
    coordinates = spline_coordinates(
        SplineCoordinatesRequest(function=revived, degree=0, smoothness=0)
    )
    assert coordinates.basis_coordinates[0].as_fraction() == expected


@pytest.mark.parametrize("public", [False, True], ids=["native", "dispatch"])
@pytest.mark.parametrize("degree", [0, 2])
@pytest.mark.parametrize("reciprocal", [False, True], ids=["numerator", "denominator"])
def test_unit_basis_coordinates_admit_canonical_boundary(
    large_integer: int, public: bool, reciprocal: bool, degree: int
) -> None:
    value = Fraction(1, large_integer) if reciprocal else Fraction(-large_integer)
    request = SplineCoordinatesRequest(
        function=_constant(value), degree=degree, smoothness=0
    )
    if public:
        result = SplineCoordinatesResult.model_validate_json(
            encode_strict_json(
                invoke_operation(
                    "polyhedral_complex.spline.coordinates.compute",
                    request.model_dump(mode="json"),
                    Catalog.open(),
                ).output
            )
        )
    else:
        result = spline_coordinates(request)
    revived = SplineCoordinatesResult.model_validate_json(
        encode_strict_json(result.model_dump(mode="json"))
    )
    assert revived.basis_coordinates == tuple(
        CanonicalRational.from_fraction(value if monomial == (0,) else Fraction(0))
        for _, monomial in revived.spline_space.coefficient_axis
    )
    assert revived.spline_space.nullspace_basis.entries == tuple(
        tuple(CanonicalRational(num=int(i == j), den=1) for j in range(degree + 1))
        for i in range(degree + 1)
    )


@pytest.mark.parametrize("reciprocal", [False, True])
@pytest.mark.parametrize("near_boundary", [False, True])
def test_uncancelled_scalar_growth_still_refuses_oversized_components(
    large_integer: int, reciprocal: bool, near_boundary: bool
) -> None:
    component = 5 * large_integer if near_boundary else large_integer
    factor = 2 if near_boundary else 10
    value = Fraction(1, component) if reciprocal else Fraction(component)
    scalar = Fraction(1, factor) if reciprocal else Fraction(factor)
    with pytest.raises(OperationResourceAdmissionError, match="scaled coefficient"):
        piecewise_polynomial_scalar_multiply(
            PiecewisePolynomialScalarMultiplicationRequest(
                function=_constant(value),
                scalar=CanonicalRational.from_fraction(scalar),
            )
        )
