"""Domain adapter for exact canonical-form operations."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.canonical_forms._models import (
    InvariantFactorEntry,
    MatrixPolynomialEvaluationRequest,
    MatrixPolynomialEvaluationResult,
    MinimalPolynomialResult,
    MonicPolynomial,
    PrimaryDecompositionResult,
    RationalCanonicalFormResult,
    SquareMatrixRequest,
)
from jacobian.math.matrices.canonical_forms._replay import _matrix_entries
from jacobian.math.matrices.canonical_forms.operations import (
    _evaluate_polynomial,
    characteristic_polynomial,
    invariant_factors,
    minimal_polynomial,
    primary_decomposition,
)
from jacobian.math.matrices.values import RationalMatrix, rational_matrix_from_fractions


def _to_monic_polynomial(coefficients: Sequence[Fraction]) -> MonicPolynomial:
    return MonicPolynomial(
        coefficients=tuple(
            CanonicalRational.from_fraction(coefficient) for coefficient in coefficients
        )
    )


def _dense_polynomial_coefficients(
    request: MatrixPolynomialEvaluationRequest,
) -> tuple[Fraction, ...]:
    degree = max(
        (term.exponents[0] for term in request.polynomial.polynomial.terms),
        default=0,
    )
    coefficients = [Fraction(0)] * (degree + 1)
    for term in request.polynomial.polynomial.terms:
        coefficients[term.exponents[0]] = term.coefficient.as_fraction()
    return tuple(coefficients)


def evaluate_matrix_polynomial_value(
    request: MatrixPolynomialEvaluationRequest,
) -> RationalMatrix:
    evaluated = _evaluate_polynomial(
        _matrix_entries(request.matrix),
        _dense_polynomial_coefficients(request),
    )
    return rational_matrix_from_fractions(evaluated)


def compute_matrix_polynomial_evaluation(
    request: MatrixPolynomialEvaluationRequest,
) -> MatrixPolynomialEvaluationResult:
    return MatrixPolynomialEvaluationResult._from_kernel(
        request=request,
        value=evaluate_matrix_polynomial_value(request),
    )


def compute_minimal_polynomial(
    request: SquareMatrixRequest,
) -> MinimalPolynomialResult:
    entries = _matrix_entries(request.matrix)
    minimal = minimal_polynomial(entries)
    characteristic = characteristic_polynomial(entries)
    return MinimalPolynomialResult._from_kernel(
        matrix=request,
        minimal_polynomial=_to_monic_polynomial(minimal),
        characteristic_polynomial=_to_monic_polynomial(characteristic),
    )


def compute_rational_canonical_form(
    request: SquareMatrixRequest,
) -> RationalCanonicalFormResult:
    entries = _matrix_entries(request.matrix)
    factors = invariant_factors(entries)
    minimal = minimal_polynomial(entries)
    characteristic = characteristic_polynomial(entries)

    invariant_entries = tuple(
        InvariantFactorEntry(
            factor=_to_monic_polynomial(coefficients),
            block_size=len(coefficients) - 1,
        )
        for coefficients in factors
    )

    return RationalCanonicalFormResult._from_kernel(
        matrix=request,
        invariant_factors=invariant_entries,
        characteristic_polynomial=_to_monic_polynomial(characteristic),
        minimal_polynomial=_to_monic_polynomial(minimal),
    )


def compute_primary_decomposition(
    request: SquareMatrixRequest,
) -> PrimaryDecompositionResult:
    entries = _matrix_entries(request.matrix)
    components = primary_decomposition(entries)
    minimal = minimal_polynomial(entries)
    return PrimaryDecompositionResult._from_kernel(
        matrix=request,
        components=tuple(
            _to_monic_polynomial(coefficient) for coefficient in components
        ),
        minimal_polynomial=_to_monic_polynomial(minimal),
    )


def verify_matrix_polynomial_evaluation_result(
    result: MatrixPolynomialEvaluationResult,
) -> bool:
    """Replay a separately supplied matrix-polynomial claim in its admitted envelope."""

    request = MatrixPolynomialEvaluationRequest(
        matrix=result.source_matrix,
        polynomial=result.polynomial,
    )
    return result.value == evaluate_matrix_polynomial_value(request)


def verify_minimal_polynomial_result(result: MinimalPolynomialResult) -> bool:
    """Check a separately supplied minimal-polynomial claim against its source."""

    from jacobian.math.matrices.canonical_forms._replay import _coefficients_of

    entries = _matrix_entries(result.matrix.matrix)
    return _coefficients_of(result.minimal_polynomial) == tuple(
        minimal_polynomial(entries)
    ) and _coefficients_of(result.characteristic_polynomial) == tuple(
        characteristic_polynomial(entries)
    )


def verify_rational_canonical_form_result(
    result: RationalCanonicalFormResult,
) -> bool:
    """Check a separately supplied invariant-factor claim against its source."""

    from jacobian.math.matrices.canonical_forms._replay import _coefficients_of

    entries = _matrix_entries(result.matrix.matrix)
    return (
        tuple(_coefficients_of(entry.factor) for entry in result.invariant_factors)
        == tuple(invariant_factors(entries))
        and _coefficients_of(result.minimal_polynomial)
        == tuple(minimal_polynomial(entries))
        and _coefficients_of(result.characteristic_polynomial)
        == tuple(characteristic_polynomial(entries))
    )


def verify_primary_decomposition_result(result: PrimaryDecompositionResult) -> bool:
    """Check a separately supplied primary-decomposition claim against its source."""

    from jacobian.math.matrices.canonical_forms._replay import _coefficients_of

    entries = _matrix_entries(result.matrix.matrix)
    return tuple(
        _coefficients_of(component) for component in result.components
    ) == tuple(primary_decomposition(entries)) and _coefficients_of(
        result.minimal_polynomial
    ) == tuple(minimal_polynomial(entries))
