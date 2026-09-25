import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.complexes import _spline as spline_kernel
from jacobian.math.geometry.polytopes.complexes._models import (
    ComplexPoint,
    PieceAssignment,
    PiecewisePolynomialResult,
    PiecewisePolynomialScalarMultiplicationRequest,
)
from jacobian.math.geometry.polytopes.complexes._spline import (
    piecewise_polynomial_evaluate,
    piecewise_polynomial_from_maximal_pieces,
    piecewise_polynomial_scalar_multiply,
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
                vertex_id=f"{prefix}{i}",
                coordinates=({"num": point, "den": 1},),
            )
            for i, point in enumerate((left, right))
        ),
    )


def _linear(coefficient: int = 1) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=coefficient, den=1),
                    exponents=(1,),
                ),
            )
        ),
    )


def _function(coefficient: int = 1):
    complex_value = polytopal_complex_closure(
        (_interval(0, 1, "a"), _interval(1, 2, "b"))
    )
    return piecewise_polynomial_from_maximal_pieces(
        complex_value,
        tuple(
            PieceAssignment(cell_id=cell.cell_id, polynomial=_linear(coefficient))
            for cell in complex_value.maximal_cells
        ),
    )


def test_scalar_multiple_matches_exact_coefficient_and_evaluation_oracles():
    function = _function()
    result = piecewise_polynomial_scalar_multiply(
        PiecewisePolynomialScalarMultiplicationRequest(
            function=function, scalar=CanonicalRational(num=3, den=2)
        )
    )

    assert result.status == "COMPATIBLE"
    assert (
        PiecewisePolynomialResult.model_validate_json(
            json.dumps(result.model_dump(mode="json"))
        )
        == result
    )
    assert all(
        piece.polynomial.polynomial.terms[0].coefficient.as_fraction() == Fraction(3, 2)
        for piece in result.pieces
    )
    for coordinate, expected in ((0, 0), (1, Fraction(3, 2)), (2, 3)):
        evaluated = piecewise_polynomial_evaluate(
            result,
            ComplexPoint(coordinates=(CanonicalRational(num=coordinate, den=1),)),
        )
        assert evaluated.value is not None
        assert evaluated.value.as_fraction() == expected

    twice_scaled = piecewise_polynomial_scalar_multiply(
        PiecewisePolynomialScalarMultiplicationRequest(
            function=result, scalar=CanonicalRational(num=2, den=1)
        )
    )
    direct = piecewise_polynomial_scalar_multiply(
        PiecewisePolynomialScalarMultiplicationRequest(
            function=function, scalar=CanonicalRational(num=3, den=1)
        )
    )
    assert twice_scaled == direct


def test_zero_scalar_returns_the_zero_function_with_the_same_exact_domain():
    function = _function()
    result = piecewise_polynomial_scalar_multiply(
        PiecewisePolynomialScalarMultiplicationRequest(
            function=function, scalar=CanonicalRational(num=0, den=1)
        )
    )

    assert result.complex == function.complex
    assert result.status == "COMPATIBLE"
    assert all(not piece.polynomial.polynomial.terms for piece in result.pieces)
    assert all(
        not row.reduced_difference.polynomial.terms for row in result.compatibility
    )


def test_scalar_multiple_recomputes_and_rejects_forged_compatibility_claim():
    function = _function()
    pieces = list(function.pieces)
    pieces[1] = PieceAssignment(
        cell_id=pieces[1].cell_id,
        polynomial=RationalPolynomial(
            variables=("x",),
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(num=2, den=1), exponents=(0,)
                    ),
                )
            ),
        ),
    )
    forged = PiecewisePolynomialResult.model_construct(
        complex=function.complex,
        pieces=tuple(pieces),
        compatibility=function.compatibility,
        status="COMPATIBLE",
        obstruction_face_id=None,
        obstruction_difference=None,
    )

    with pytest.raises(OperationDomainValidationError, match="continuity claims"):
        piecewise_polynomial_scalar_multiply(
            PiecewisePolynomialScalarMultiplicationRequest(
                function=forged, scalar=CanonicalRational(num=2, den=1)
            )
        )


def test_catalog_example_executes_through_the_public_typed_operation():
    catalog = Catalog.open()
    operation = catalog.operation("piecewise_polynomial.scalar_multiply.compute")
    assert operation is not None and operation.examples

    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    validated = PiecewisePolynomialResult.model_validate_json(json.dumps(result.output))
    assert validated.status == "COMPATIBLE"
    assert (
        validated.pieces[0].polynomial.polynomial.terms[0].coefficient.as_fraction()
        == 2
    )


def test_scalar_with_two_maximal_components_is_admitted():
    function = _function(coefficient=1)
    numerator = 10**20_000 + 1
    denominator = 10**20_000 + 3
    scalar = CanonicalRational.from_integer_ratio(numerator, denominator)

    result = piecewise_polynomial_scalar_multiply(
        PiecewisePolynomialScalarMultiplicationRequest(function=function, scalar=scalar)
    )

    coefficient = result.pieces[0].polynomial.polynomial.terms[0].coefficient
    assert coefficient.as_fraction() == scalar.as_fraction()


def test_scalar_growth_is_rejected_before_coefficient_expansion(monkeypatch):
    function = _function(coefficient=10)
    monkeypatch.setattr(spline_kernel, "MAX_CANONICAL_RATIONAL_DIGITS", 3)

    with pytest.raises(OperationResourceAdmissionError, match="scaled coefficient"):
        piecewise_polynomial_scalar_multiply(
            PiecewisePolynomialScalarMultiplicationRequest(
                function=function, scalar=CanonicalRational(num=10, den=1)
            )
        )
