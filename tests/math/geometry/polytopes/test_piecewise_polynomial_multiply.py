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
from jacobian.math.geometry.polytopes.complexes._models import (
    ComplexPoint,
    PieceAssignment,
    PiecewisePolynomialMultiplicationRequest,
    PiecewisePolynomialResult,
)
from jacobian.math.geometry.polytopes.complexes._spline import (
    piecewise_polynomial_evaluate,
    piecewise_polynomial_from_maximal_pieces,
    piecewise_polynomial_multiply,
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


def _poly(coefficients: dict[tuple[int, ...], Fraction | int]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(value)),
                    exponents=exponents,
                )
                for exponents, value in sorted(coefficients.items(), reverse=True)
                if value
            )
        ),
    )


def _complex():
    return polytopal_complex_closure((_interval(0, 1, "a"), _interval(1, 2, "b")))


def _function(complex_value, coefficients):
    pieces = tuple(
        PieceAssignment(cell_id=cell.cell_id, polynomial=_poly(coefficients))
        for cell in complex_value.maximal_cells
    )
    return piecewise_polynomial_from_maximal_pieces(complex_value, pieces)


def _coefficient_map(polynomial: RationalPolynomial):
    return {
        term.exponents: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }


def test_piecewise_product_is_exact_and_compatible_on_shared_face():
    complex_value = _complex()
    left = _function(complex_value, {(1,): Fraction(1, 2), (0,): 1})
    right = _function(complex_value, {(1,): 1, (0,): 1})

    result = piecewise_polynomial_multiply(
        PiecewisePolynomialMultiplicationRequest(left=left, right=right)
    )

    expected = {
        (2,): Fraction(1, 2),
        (1,): Fraction(3, 2),
        (0,): Fraction(1),
    }
    assert result.status == "COMPATIBLE"
    assert all(row.compatible for row in result.compatibility)
    assert all(
        not row.reduced_difference.polynomial.terms for row in result.compatibility
    )
    assert all(
        _coefficient_map(piece.polynomial) == expected for piece in result.pieces
    )
    evaluation = piecewise_polynomial_evaluate(
        result, ComplexPoint(coordinates=(CanonicalRational(num=1, den=1),))
    )
    assert evaluation.value is not None
    assert evaluation.value.as_fraction() == 3


def test_piecewise_product_rejects_forged_compatible_claim():
    complex_value = _complex()
    forged = PiecewisePolynomialResult.model_construct(
        complex=complex_value,
        pieces=tuple(
            PieceAssignment(
                cell_id=cell.cell_id,
                polynomial=_poly({(0,): 1 if cell.cell_id == "M0" else 2}),
            )
            for cell in complex_value.maximal_cells
        ),
        compatibility=(),
        status="COMPATIBLE",
        obstruction_face_id=None,
        obstruction_difference=None,
    )
    valid = _function(complex_value, {(0,): 1})

    with pytest.raises(OperationDomainValidationError, match="compatibility claims"):
        piecewise_polynomial_multiply(
            PiecewisePolynomialMultiplicationRequest(left=forged, right=valid)
        )


def test_piecewise_product_preflights_aggregate_convolution_work():
    complex_value = polytopal_complex_closure((_interval(0, 1, "a"),))
    terms = tuple(
        RationalPolynomialTerm(
            coefficient=CanonicalRational(num=1, den=1), exponents=(exponent,)
        )
        for exponent in range(1023, -1, -1)
    )
    polynomial = RationalPolynomial(
        variables=("x",), polynomial=SparseRationalPolynomial(terms=terms)
    )
    function = PiecewisePolynomialResult(
        complex=complex_value,
        pieces=(PieceAssignment(cell_id="M0", polynomial=polynomial),),
        compatibility=(),
        status="COMPATIBLE",
    )

    with pytest.raises(OperationResourceAdmissionError, match="aggregate piecewise"):
        piecewise_polynomial_multiply(
            PiecewisePolynomialMultiplicationRequest(left=function, right=function)
        )


def test_catalog_piecewise_multiplication_example_executes():
    catalog = Catalog.open()
    operation = catalog.operation("piecewise_polynomial.multiply.compute")
    assert operation is not None and operation.examples
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    validated = PiecewisePolynomialResult.model_validate_json(json.dumps(result.output))
    assert validated.status == "COMPATIBLE"
    assert _coefficient_map(validated.pieces[0].polynomial) == {
        (2,): Fraction(1),
        (1,): Fraction(1),
    }
