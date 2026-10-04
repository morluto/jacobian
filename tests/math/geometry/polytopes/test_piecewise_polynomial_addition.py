from collections.abc import Mapping
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
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
    PolytopalComplexClosureResult,
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
                coordinates=(CanonicalRational(num=point, den=1),),
            )
            for i, point in enumerate((left, right))
        ),
    )


def _poly(coefficients: Mapping[tuple[int, ...], int]) -> RationalPolynomial:
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


def _complex() -> PolytopalComplexClosureResult:
    return polytopal_complex_closure((_interval(0, 1, "a"), _interval(1, 2, "b")))


def _function(
    complex_value: PolytopalComplexClosureResult,
    coefficients: Mapping[tuple[int, ...], int],
) -> PiecewisePolynomialResult:
    pieces = tuple(
        PieceAssignment(cell_id=cell.cell_id, polynomial=_poly(coefficients))
        for cell in complex_value.maximal_cells
    )
    return piecewise_polynomial_from_maximal_pieces(complex_value, pieces)


def test_addition_matches_independent_coefficient_oracle_and_composes_at_shared_point() -> (
    None
):
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


def _coefficient_map(polynomial: RationalPolynomial) -> dict[tuple[int, ...], Fraction]:
    return {
        term.exponents: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }


def test_addition_rejects_different_complex_values_even_when_support_matches() -> None:
    left_complex = polytopal_complex_closure((_interval(0, 2, "a"),))
    right_complex = _complex()
    left = _function(left_complex, {(0,): 1})
    right = _function(right_complex, {(0,): 1})

    with pytest.raises(OperationDomainValidationError) as exc_info:
        piecewise_polynomial_add(
            PiecewisePolynomialAdditionRequest(left=left, right=right)
        )
    assert exc_info.value.errors()[0]["type"] == "polytopal_complex.addition_complex"


def test_addition_rejects_a_claimed_piecewise_function_that_is_discontinuous() -> None:
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

    with pytest.raises(OperationDomainValidationError) as exc_info:
        piecewise_polynomial_add(
            PiecewisePolynomialAdditionRequest(left=discontinuous, right=compatible)
        )
    assert exc_info.value.errors()[0]["type"] == "polytopal_complex.addition_continuity"


def test_addition_preflights_union_term_count_before_coefficient_arithmetic() -> None:
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

    def value(terms: tuple[RationalPolynomialTerm, ...]) -> PiecewisePolynomialResult:
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

    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        piecewise_polynomial_add(
            PiecewisePolynomialAdditionRequest(
                left=value(left_terms), right=value(right_terms)
            )
        )
    assert exc_info.value.errors()[0]["type"] == "polytopal_complex.addition_terms"


def test_addition_admits_result_support_after_exact_cancellation() -> None:
    complex_value = polytopal_complex_closure((_interval(0, 1, "a"),))
    shared_terms = 3_000
    left_terms: dict[tuple[int, ...], int] = {
        (exponent,): 1 for exponent in range(MAX_POLYNOMIAL_TERMS)
    }
    right_terms: dict[tuple[int, ...], int] = {
        **{(exponent,): -1 for exponent in range(shared_terms)},
        **{
            (exponent,): 1
            for exponent in range(MAX_POLYNOMIAL_TERMS, MAX_POLYNOMIAL_TERMS + 1_096)
        },
    }

    result = piecewise_polynomial_add(
        PiecewisePolynomialAdditionRequest(
            left=_function(complex_value, left_terms),
            right=_function(complex_value, right_terms),
        )
    )

    output_terms = result.pieces[0].polynomial.polynomial.terms
    assert len(output_terms) == 2 * (MAX_POLYNOMIAL_TERMS - shared_terms)


def test_addition_admits_exact_cancellation_before_the_coefficient_growth_bound() -> (
    None
):
    modulus = 10**20_000 + 3

    def constant(numerator: int) -> PiecewisePolynomialResult:
        complex_value = polytopal_complex_closure((_interval(0, 1, "a"),))
        pieces = tuple(
            PieceAssignment(
                cell_id=cell.cell_id,
                polynomial=RationalPolynomial(
                    variables=("x",),
                    polynomial=SparseRationalPolynomial(
                        terms=(
                            RationalPolynomialTerm(
                                coefficient=CanonicalRational(
                                    num=numerator, den=modulus
                                ),
                                exponents=(0,),
                            ),
                        )
                    ),
                ),
            )
            for cell in complex_value.maximal_cells
        )
        return piecewise_polynomial_from_maximal_pieces(complex_value, pieces)

    result = piecewise_polynomial_add(
        PiecewisePolynomialAdditionRequest(left=constant(1), right=constant(-1))
    )
    assert result.status == "COMPATIBLE"
    assert all(_coefficient_map(piece.polynomial) == {} for piece in result.pieces)


def test_native_addition_rejects_a_forged_request_with_a_typed_error() -> None:
    with pytest.raises(OperationDomainValidationError) as exc_info:
        piecewise_polynomial_add(PiecewisePolynomialAdditionRequest.model_construct())
    assert exc_info.value.errors()[0]["type"] == "polytopal_complex.addition_type"
