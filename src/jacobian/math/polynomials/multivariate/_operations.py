"""Domain-owned multivariate polynomial operations over ``QQ``."""

from __future__ import annotations

from typing import Any, Literal

from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.multivariate import _factor_backend
from jacobian.math.polynomials.multivariate._division import (
    MultivariateDivisionRequest,
    MultivariateDivisionResult,
)
from jacobian.math.polynomials.multivariate._factor_backend import (
    FactorBackendExhaustedError,
    FactorBackendFailureError,
    FactorBackendInterruptedError,
)
from jacobian.math.polynomials.multivariate._gcd import (
    MultivariateGcdRequest,
    MultivariateGcdResult,
)
from jacobian.math.polynomials.multivariate._models import (
    _MAX_FACTOR_OUTPUT_TERMS as _MAX_OUTPUT_TERMS,
)
from jacobian.math.polynomials.multivariate._models import (
    _MAX_SUBRESULTANT_SEQUENCE_TERMS,
    MultivariateFactorRequest,
    MultivariateFactorResult,
    MultivariateIrreducibleFactor,
    MultivariatePrincipalSubresultantCoefficient,
    MultivariateSubresultantMember,
    MultivariateSubresultantSequenceRequest,
    MultivariateSubresultantSequenceResult,
    _degree_in_variable,
    _monic_content_fraction,
)
from jacobian.math.polynomials.multivariate._resultant import (
    MultivariateResultantRequest,
    MultivariateResultantResult,
    _sylvester_resultant_value,
)
from jacobian.math.polynomials.values import RationalPolynomial


class MultivariateOutputBudgetError(RuntimeError):
    """A valid computation produced more output than its public contract permits."""


def _result_polynomial(
    poly: Any,
    variables: tuple[str, ...],
) -> Any:
    """Convert a SymPy ``Poly`` to a ``RationalPolynomial``, re-raising budget errors."""

    try:
        return rational_polynomial_from_sympy(
            poly,
            variables,
            maximum_terms=_MAX_OUTPUT_TERMS,
        )
    except ValueError as exc:
        if "term operation budget" in str(exc):
            raise MultivariateOutputBudgetError(str(exc)) from exc
        raise


def compute_multivariate_gcd(request: MultivariateGcdRequest) -> MultivariateGcdResult:
    """Compute the GCD of two multivariate polynomials over ``QQ``."""

    left = rational_polynomial_to_sympy(request.left)
    right = rational_polynomial_to_sympy(request.right)
    gcd = left.gcd(right)
    return MultivariateGcdResult(
        gcd=_result_polynomial(gcd, request.left.variables),
    )


def compute_multivariate_division(
    request: MultivariateDivisionRequest,
) -> MultivariateDivisionResult:
    """Divide one multivariate polynomial by another with a declared monomial order."""

    variables = request.left.variables
    symbols = symbols_for_variables(variables)

    # Build a low-level ring with the requested monomial order so that
    # multivariate division respects the declared term ordering.  The
    # ``Poly`` API uses the implicit lex order and does not expose the
    # monomial-order keyword in its constructor.
    from sympy import QQ
    from sympy.polys.rings import ring as sympy_ring

    ring_obj = sympy_ring(symbols, QQ, request.monomial_order)[0]

    left_poly = rational_polynomial_to_sympy(request.left)
    right_poly = rational_polynomial_to_sympy(request.right)

    left_ring = ring_obj.from_dict(
        {exponents: coeff for exponents, coeff in left_poly.terms()}  # noqa: C416
    )
    right_ring = ring_obj.from_dict(
        {exponents: coeff for exponents, coeff in right_poly.terms()}  # noqa: C416
    )

    quotient_ring, remainder_ring = left_ring.div(right_ring)

    # Convert back to ``Poly`` for the canonical sparse wire contract.
    quotient_poly = _to_poly(quotient_ring, symbols)
    remainder_poly = _to_poly(remainder_ring, symbols)

    # Verify the exact reconstruction: left == quotient * right + remainder.
    reconstruction = quotient_poly * right_poly + remainder_poly
    if reconstruction != left_poly:
        raise MultivariateOutputBudgetError(
            "multivariate division reconstruction failed"
        )

    return MultivariateDivisionResult(
        quotient=_result_polynomial(quotient_poly, variables),
        remainder=_result_polynomial(remainder_poly, variables),
        monomial_order=request.monomial_order,
    )


def compute_multivariate_resultant(
    request: MultivariateResultantRequest,
) -> MultivariateResultantResult:
    """Compute the resultant of two multivariate polynomials w.r.t. one variable."""
    return MultivariateResultantResult(
        left=request.left,
        right=request.right,
        elimination_variable=request.elimination_variable,
        resultant=_sylvester_resultant_value(request),
    )


def compute_multivariate_subresultant_sequence(
    request: MultivariateSubresultantSequenceRequest,
) -> MultivariateSubresultantSequenceResult:
    """Return the exact nonzero Brown PRS in one declared main variable."""

    from jacobian.math.polynomials.multivariate._subresultants import (
        polynomial_leading_coefficient_in_remaining_ring,
        polynomial_resultant_in_remaining_ring,
        polynomial_subresultant_sequence,
    )

    source_order, polynomials, principal_coefficients = (
        polynomial_subresultant_sequence(
            request.left,
            request.right,
            request.main_variable,
            maximum_terms=_MAX_SUBRESULTANT_SEQUENCE_TERMS,
        )
    )
    variable_index = request.left.variables.index(request.main_variable)
    members = tuple(
        MultivariateSubresultantMember(
            polynomial=polynomial,
            degree_in_main_variable=_degree_in_variable(polynomial, variable_index),
        )
        for polynomial in polynomials
    )
    degrees = {member.degree_in_main_variable for member in members}
    greatest_degree = max(degrees)
    resultant = polynomial_resultant_in_remaining_ring(
        request.left,
        request.right,
        request.main_variable,
        maximum_terms=_MAX_SUBRESULTANT_SEQUENCE_TERMS,
    )
    gcd_degree = members[-1].degree_in_main_variable
    gcd_leading_coefficient = polynomial_leading_coefficient_in_remaining_ring(
        members[-1].polynomial,
        request.main_variable,
        maximum_terms=_MAX_SUBRESULTANT_SEQUENCE_TERMS,
    )
    left_degree = _degree_in_variable(request.left, variable_index)
    right_degree = _degree_in_variable(request.right, variable_index)
    return MultivariateSubresultantSequenceResult(
        left=request.left,
        right=request.right,
        main_variable=request.main_variable,
        source_order=source_order,
        members=members,
        skipped_member_degrees=tuple(
            degree for degree in range(greatest_degree + 1) if degree not in degrees
        ),
        principal_subresultant_coefficients=tuple(
            MultivariatePrincipalSubresultantCoefficient(
                index=index,
                coefficient=coefficient,
            )
            for index, coefficient in enumerate(principal_coefficients)
        ),
        resultant=resultant,
        resultant_sign_from_sequence_order=(
            -1
            if source_order == "RIGHT_LEFT" and left_degree * right_degree % 2 == 1
            else 1
        ),
        gcd_member_index=len(members) - 1,
        gcd_degree_in_main_variable=gcd_degree,
        gcd_member_leading_coefficient=gcd_leading_coefficient,
    )


def _to_poly(ring_element: Any, symbols: tuple[Any, ...]) -> Any:
    """Convert a low-level ring element to a SymPy ``Poly`` in ``QQ``."""

    from sympy import QQ, Poly

    return Poly(ring_element.as_expr(), *symbols, domain=QQ)


def _sympy_factorization(
    polynomial: RationalPolynomial,
    *,
    wall_seconds: float | None = None,
) -> tuple[Any, tuple[tuple[Any, int], ...], Any]:
    """Run the bounded killable ``factor_list`` backend and verify reconstruction."""

    from jacobian.math.polynomials._sympy import _monic_decomposition

    decomposition = _factor_backend.run_bounded_factorization(
        polynomial,
        wall_seconds=wall_seconds,
    )
    source = rational_polynomial_to_sympy(polynomial)
    return _monic_decomposition(
        source,
        decomposition,
        label="multivariate factorization",
    )


def multivariate_factor(request: MultivariateFactorRequest) -> MultivariateFactorResult:
    """Exact factorization over ``QQ[variables]`` via SymPy's ``factor_list``.

    The factorization kernel runs in a bounded, killable worker process.
    When the exact factorization contains an irreducible factor beyond the
    public output-term budget, or its serialized form exceeds the declared
    transport output bound, returns the typed ``OUTPUT_BUDGET_EXCEEDED``
    outcome instead of a host exception; ``coefficient`` then carries the
    exact positive rational content of the restated source polynomial.  A
    worker stopped by its deadline or cancellation, killed by an enforced
    resource cap such as its CPU or address-space budget, crashed, or
    running without containment establishes nothing about output size, so
    it returns the distinct non-mathematical ``EXECUTION_FAILED`` outcome
    instead.
    """

    from jacobian._exact import CanonicalRational

    def _bounded_outcome(
        status: Literal["OUTPUT_BUDGET_EXCEEDED", "EXECUTION_FAILED"],
    ) -> MultivariateFactorResult:
        # Non-FACTORIZED outcomes restate the source and carry its exact
        # positive primitive content, matching result validation.
        return MultivariateFactorResult(
            status=status,
            coefficient=CanonicalRational.from_fraction(
                _factor_backend.primitive_content_fraction(request.polynomial)
            ),
            factors=(),
            reconstructed=request.polynomial,
            normalization=None,
            product_reconstruction=None,
        )

    try:
        coefficient, raw_factors, reconstructed = _sympy_factorization(
            request.polynomial
        )
    except FactorBackendInterruptedError:
        return _bounded_outcome("EXECUTION_FAILED")
    except FactorBackendExhaustedError:
        return _bounded_outcome("OUTPUT_BUDGET_EXCEEDED")
    except FactorBackendFailureError:
        return _bounded_outcome("EXECUTION_FAILED")
    coefficient_value = CanonicalRational.from_fraction(
        _monic_content_fraction(coefficient)
    )

    factors_list = []
    try:
        for factor, multiplicity in raw_factors:
            factors_list.append(
                MultivariateIrreducibleFactor(
                    factor=_result_polynomial(factor, request.polynomial.variables),
                    multiplicity=multiplicity,
                )
            )
    except MultivariateOutputBudgetError:
        # The oversized-factor branch reports the same exact positive
        # content as every other bounded outcome; the signed monic leading
        # coefficient is a FACTORIZED-only convention tied to its factors.
        return _bounded_outcome("OUTPUT_BUDGET_EXCEEDED")
    factors_list.sort(
        key=lambda record: (
            record.multiplicity,
            max(
                (sum(term.exponents) for term in record.factor.polynomial.terms),
                default=0,
            ),
            tuple(
                (term.exponents, term.coefficient.num, term.coefficient.den)
                for term in record.factor.polynomial.terms
            ),
        )
    )
    factors = tuple(factors_list)

    reconstructed_poly = _result_polynomial(reconstructed, request.polynomial.variables)

    return MultivariateFactorResult(
        coefficient=coefficient_value,
        factors=factors,
        reconstructed=reconstructed_poly,
    )
