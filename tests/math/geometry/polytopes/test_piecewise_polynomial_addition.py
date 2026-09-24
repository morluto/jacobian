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
    PiecewisePolynomialAdditionRequest,
    PiecewisePolynomialResult,
)
from jacobian.math.geometry.polytopes.complexes._spline import (
    piecewise_polynomial_add,
    piecewise_polynomial_evaluate,
    piecewise_polynomial_from_maximal_pieces,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    polytopal_complex_closure,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_TERMS,
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


def _poly(coefficients: dict[tuple[int, ...], int]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=value, den=1),
                    exponents=exponents,
                )
                for exponents, value in sorted(coefficients.items(), reverse=True)
                if value
            )
        ),
    )


def _complex():
    return polytopal_complex_closure((_interval(0, 1, "a"), _interval(1, 2, "b")))


def _function(complex_value, coefficients: dict[tuple[int, ...], int]):
    pieces = tuple(
        PieceAssignment(cell_id=cell.cell_id, polynomial=_poly(coefficients))
        for cell in complex_value.maximal_cells
    )
    return piecewise_polynomial_from_maximal_pieces(complex_value, pieces)


def _coefficient_map(polynomial: RationalPolynomial) -> dict[tuple[int, ...], Fraction]:
    return {
        term.exponents: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }


def test_addition_matches_independent_coefficient_oracle_and_composes_at_shared_point():
    complex_value = _complex()
    left = _function(complex_value, {(1,): 1})
    right = _function(complex_value, {(2,): 1, (0,): 1})

    result = piecewise_polynomial_add(
        PiecewisePolynomialAdditionRequest(left=left, right=right)
    )

    expected = {(2,): Fraction(1), (1,): Fraction(1), (0,): Fraction(1)}
    assert result.status == "COMPATIBLE"
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


def test_addition_rejects_different_complex_values_even_when_support_matches():
    left_complex = polytopal_complex_closure((_interval(0, 2, "a"),))
    right_complex = _complex()
    left = _function(left_complex, {(0,): 1})
    right = _function(right_complex, {(0,): 1})

    with pytest.raises(
        OperationDomainValidationError, match="identical canonical complex"
    ):
        piecewise_polynomial_add(
            PiecewisePolynomialAdditionRequest(left=left, right=right)
        )


def test_addition_rejects_a_claimed_piecewise_function_that_is_discontinuous():
    complex_value = _complex()
    pieces = tuple(
        PieceAssignment(
            cell_id=cell.cell_id,
            polynomial=_poly({(0,): 1 if cell.cell_id == "M0" else 2}),
        )
        for cell in complex_value.maximal_cells
    )
    discontinuous = piecewise_polynomial_from_maximal_pieces(complex_value, pieces)
    compatible = _function(complex_value, {(0,): 1})

    with pytest.raises(OperationDomainValidationError, match="only compatible C0"):
        piecewise_polynomial_add(
            PiecewisePolynomialAdditionRequest(left=discontinuous, right=compatible)
        )


def test_addition_preflights_union_term_count_before_coefficient_arithmetic():
    complex_value = polytopal_complex_closure((_interval(0, 1, "a"),))
    left_terms = tuple(
        RationalPolynomialTerm(
            coefficient=CanonicalRational(num=1, den=1), exponents=(exponent,)
        )
        for exponent in range(2 * MAX_POLYNOMIAL_TERMS - 2, -1, -2)
    )
    right_terms = tuple(
        RationalPolynomialTerm(
            coefficient=CanonicalRational(num=1, den=1), exponents=(exponent,)
        )
        for exponent in range(2 * MAX_POLYNOMIAL_TERMS - 1, 0, -2)
    )

    def value(terms):
        return PiecewisePolynomialResult(
            complex=complex_value,
            pieces=(
                PieceAssignment(
                    cell_id="M0",
                    polynomial=RationalPolynomial(
                        variables=("x",),
                        polynomial=SparseRationalPolynomial(terms=terms),
                    ),
                ),
            ),
            compatibility=(),
            status="COMPATIBLE",
        )

    with pytest.raises(OperationResourceAdmissionError, match="sum piece may contain"):
        piecewise_polynomial_add(
            PiecewisePolynomialAdditionRequest(
                left=value(left_terms), right=value(right_terms)
            )
        )


def test_catalog_addition_example_executes_through_typed_contract():
    catalog = Catalog.open()
    operation = catalog.operation("piecewise_polynomial.add.compute")
    assert operation is not None and operation.examples
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    validated = PiecewisePolynomialResult.model_validate_json(json.dumps(result.output))
    assert validated.status == "COMPATIBLE"
    assert _coefficient_map(validated.pieces[0].polynomial) == {(0,): Fraction(3)}
