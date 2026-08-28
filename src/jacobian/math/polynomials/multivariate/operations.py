"""Domain-owned multivariate polynomial operations over ``QQ``."""

from __future__ import annotations

from typing import Any, Literal

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.multivariate import _factor_backend
from jacobian.math.polynomials.multivariate._division import (
    MultivariateDivisionResult,
)
from jacobian.math.polynomials.multivariate._factor_backend import (
    FactorBackendExhaustedError,
    FactorBackendFailureError,
    FactorBackendInterruptedError,
)
from jacobian.math.polynomials.multivariate._factor_models import (
    _MAX_FACTOR_OUTPUT_TERMS,
    MultivariateFactorResult,
    MultivariateIrreducibleFactor,
)
from jacobian.math.polynomials.multivariate._gcd import (
    MultivariateGcdResult,
)
from jacobian.math.polynomials.multivariate._models import (
    _MAX_ELIMINATION_DEGREE_SUM,
    _MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
    _MAX_MULTIVARIATE_EXPONENT,
    _MAX_MULTIVARIATE_TERMS,
    _MULTIVARIATE_MIN_VARIABLES,
    _degree_in_variable,
    _validate_multivariate_pair,
    _validation_error,
)
from jacobian.math.polynomials.multivariate._resultant import (
    _MAX_RESULTANT_TERMS,
    MultivariateResultantResult,
    _resultant_support_bound,
    _sylvester_resultant_value,
)
from jacobian.math.polynomials.multivariate._subresultants import (
    _MAX_SUBRESULTANT_ARITHMETIC_TERM_PAIRS,
    _MAX_SUBRESULTANT_COEFFICIENT_BITS,
    _MAX_SUBRESULTANT_COEFFICIENT_SUPPORT,
    _MAX_SUBRESULTANT_INTERMEDIATE_COEFFICIENT_BITS,
    _MAX_SUBRESULTANT_SEQUENCE_TERMS,
    _MAX_SUBRESULTANT_SERIALIZED_COEFFICIENT_BITS,
    MultivariatePrincipalSubresultantCoefficient,
    MultivariateSubresultantMember,
    MultivariateSubresultantSequenceResult,
    _subresultant_envelope,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    require_polynomial_budget,
)

__all__ = [
    "multivariate_division",
    "multivariate_factor",
    "multivariate_gcd",
    "multivariate_resultant",
    "multivariate_subresultant_sequence",
]


def _run_admission(admission: Any) -> None:
    try:
        admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=(), code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=(), code="polynomial.multivariate_admission", message=str(exc)
        ) from exc


def _admit_pair(left: RationalPolynomial, right: RationalPolynomial) -> None:
    _validate_multivariate_pair(left, right)
    for polynomial in (left, right):
        require_polynomial_budget(
            polynomial,
            maximum_terms=_MAX_MULTIVARIATE_TERMS,
            maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
            maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
        )


def _admit_gcd(left: RationalPolynomial, right: RationalPolynomial) -> None:
    _admit_pair(left, right)


def _admit_division(left: RationalPolynomial, right: RationalPolynomial) -> None:
    _admit_pair(left, right)
    if not right.polynomial.terms:
        raise _validation_error("divisor polynomial must be nonzero")


def _admit_factor(polynomial: RationalPolynomial) -> None:
    if len(polynomial.variables) < _MULTIVARIATE_MIN_VARIABLES:
        raise _validation_error(
            "multivariate factorization requires at least two variables; "
            "univariate polynomials are handled by polynomial.factor.compute"
        )
    if not polynomial.polynomial.terms:
        raise _validation_error("zero polynomial has no factorization")
    require_polynomial_budget(
        polynomial,
        maximum_terms=_MAX_MULTIVARIATE_TERMS,
        maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
        maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
    )
    from jacobian.math.polynomials.multivariate._factor_models import (
        _require_representable_content,
    )

    _require_representable_content(polynomial)


def _admit_resultant(
    left: RationalPolynomial, right: RationalPolynomial, elimination_variable: str
) -> None:
    _admit_pair(left, right)
    if elimination_variable not in left.variables:
        raise _validation_error("elimination variable must belong to the declared ring")
    index = left.variables.index(elimination_variable)
    if (
        _degree_in_variable(left, index) + _degree_in_variable(right, index)
        > _MAX_ELIMINATION_DEGREE_SUM
    ):
        raise _validation_error("Sylvester degree exceeds the resultant budget")
    if _resultant_support_bound(left, right, index) > _MAX_RESULTANT_TERMS:
        raise _validation_error("resultant output exceeds the term budget")


def _admit_subresultants(
    left: RationalPolynomial, right: RationalPolynomial, main_variable: str
) -> None:
    _admit_pair(left, right)
    if main_variable not in left.variables:
        raise _validation_error("main variable must belong to the declared ring")
    index = left.variables.index(main_variable)
    degrees = (_degree_in_variable(left, index), _degree_in_variable(right, index))
    if any(degree == 0 for degree in degrees):
        raise _validation_error(
            "both polynomials must have positive main-variable degree"
        )
    if sum(degrees) > _MAX_ELIMINATION_DEGREE_SUM:
        raise _validation_error(
            "Sylvester order exceeds the subresultant backend budget"
        )
    envelope = _subresultant_envelope(left, right, index)
    if envelope.aggregate_terms > _MAX_SUBRESULTANT_SEQUENCE_TERMS:
        raise _validation_error(
            "formal subresultant sequence support exceeds the aggregate result-term budget"
        )
    if envelope.maximum_coefficient_support > _MAX_SUBRESULTANT_COEFFICIENT_SUPPORT:
        raise _validation_error(
            "subresultant coefficient support exceeds the intermediate polynomial-term budget"
        )
    if envelope.arithmetic_term_pairs > _MAX_SUBRESULTANT_ARITHMETIC_TERM_PAIRS:
        raise _validation_error(
            "subresultant pseudo-remainder arithmetic exceeds the term-pair work budget"
        )
    if envelope.coefficient_bits > _MAX_SUBRESULTANT_COEFFICIENT_BITS:
        raise _validation_error(
            "subresultant determinant coefficient height exceeds the exact coefficient-bit budget"
        )
    if (
        envelope.intermediate_coefficient_bits
        > _MAX_SUBRESULTANT_INTERMEDIATE_COEFFICIENT_BITS
    ):
        raise _validation_error(
            "Brown pseudo-remainder intermediate coefficient height exceeds the exact coefficient-bit budget"
        )
    if (
        envelope.serialized_coefficient_bits
        > _MAX_SUBRESULTANT_SERIALIZED_COEFFICIENT_BITS
    ):
        raise _validation_error(
            "subresultant sequence exceeds the aggregate exact-output budget"
        )


class MultivariateOutputBudgetError(RuntimeError):
    """A valid computation produced more output than its public contract permits."""


def _result_polynomial(
    poly: Any,
    variables: tuple[str, ...],
    *,
    maximum_terms: int = 1_024,
) -> Any:
    """Convert a SymPy ``Poly`` to a ``RationalPolynomial``, re-raising budget errors."""

    try:
        return rational_polynomial_from_sympy(
            poly,
            variables,
            maximum_terms=maximum_terms,
        )
    except ValueError as exc:
        if "term operation budget" in str(exc):
            raise MultivariateOutputBudgetError(str(exc)) from exc
        raise


def multivariate_gcd(
    left: RationalPolynomial, right: RationalPolynomial
) -> MultivariateGcdResult:
    """Compute the GCD of two multivariate polynomials over ``QQ``."""

    _run_admission(lambda: _admit_gcd(left, right))

    left_value = rational_polynomial_to_sympy(left)
    right_value = rational_polynomial_to_sympy(right)
    gcd = left_value.gcd(right_value)
    return MultivariateGcdResult(
        gcd=_result_polynomial(gcd, left.variables),
    )


def multivariate_division(
    left: RationalPolynomial,
    right: RationalPolynomial,
    monomial_order: Literal["lex", "grlex", "grevlex"] = "lex",
) -> MultivariateDivisionResult:
    """Divide one multivariate polynomial by another with a declared monomial order."""

    _run_admission(lambda: _admit_division(left, right))

    variables = left.variables
    symbols = symbols_for_variables(variables)

    # Build a low-level ring with the requested monomial order so that
    # multivariate division respects the declared term ordering.  The
    # ``Poly`` API uses the implicit lex order and does not expose the
    # monomial-order keyword in its constructor.
    from sympy import QQ
    from sympy.polys.rings import ring as sympy_ring

    ring_obj = sympy_ring(symbols, QQ, monomial_order)[0]

    left_poly = rational_polynomial_to_sympy(left)
    right_poly = rational_polynomial_to_sympy(right)

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
        monomial_order=monomial_order,
    )


def multivariate_resultant(
    left: RationalPolynomial,
    right: RationalPolynomial,
    elimination_variable: str,
) -> MultivariateResultantResult:
    """Compute the resultant of two multivariate polynomials w.r.t. one variable."""
    _run_admission(lambda: _admit_resultant(left, right, elimination_variable))
    return MultivariateResultantResult._from_kernel(
        left,
        right,
        elimination_variable,
        resultant=_sylvester_resultant_value(left, right, elimination_variable),
    )


def multivariate_subresultant_sequence(
    left: RationalPolynomial,
    right: RationalPolynomial,
    main_variable: str,
) -> MultivariateSubresultantSequenceResult:
    """Return the exact nonzero Brown PRS in one declared main variable."""

    _run_admission(lambda: _admit_subresultants(left, right, main_variable))

    from jacobian.math.polynomials.multivariate._subresultants import (
        polynomial_leading_coefficient_in_remaining_ring,
        polynomial_resultant_in_remaining_ring,
        polynomial_subresultant_sequence,
    )

    source_order, polynomials, principal_coefficients = (
        polynomial_subresultant_sequence(
            left,
            right,
            main_variable,
            maximum_terms=_MAX_SUBRESULTANT_SEQUENCE_TERMS,
        )
    )
    variable_index = left.variables.index(main_variable)
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
        left,
        right,
        main_variable,
        maximum_terms=_MAX_SUBRESULTANT_SEQUENCE_TERMS,
    )
    gcd_degree = members[-1].degree_in_main_variable
    gcd_leading_coefficient = polynomial_leading_coefficient_in_remaining_ring(
        members[-1].polynomial,
        main_variable,
        maximum_terms=_MAX_SUBRESULTANT_SEQUENCE_TERMS,
    )
    left_degree = _degree_in_variable(left, variable_index)
    right_degree = _degree_in_variable(right, variable_index)
    return MultivariateSubresultantSequenceResult._from_kernel(
        left,
        right,
        main_variable,
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


_SympyFactorKey = tuple[tuple[tuple[int, ...], int, int], ...]


def _monic_content_fraction(content: Any) -> Any:
    """Extract exact rational content from a monic decomposition."""

    from fractions import Fraction

    leading = getattr(content, "LC", None)
    value = leading() if callable(leading) else content
    return Fraction(int(value.p), int(value.q))


def multivariate_factor(polynomial: RationalPolynomial) -> MultivariateFactorResult:
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

    _run_admission(lambda: _admit_factor(polynomial))

    from jacobian._exact import CanonicalRational

    def _bounded_outcome(
        status: Literal["OUTPUT_BUDGET_EXCEEDED", "EXECUTION_FAILED"],
    ) -> MultivariateFactorResult:
        # Non-FACTORIZED outcomes restate the source and carry its exact
        # positive primitive content, matching result validation.
        return MultivariateFactorResult._from_kernel(
            status=status,
            coefficient=CanonicalRational.from_fraction(
                _factor_backend.primitive_content_fraction(polynomial)
            ),
            factors=(),
            reconstructed=polynomial,
        )

    try:
        coefficient, raw_factors, reconstructed = _sympy_factorization(polynomial)
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
                    factor=_result_polynomial(
                        factor,
                        polynomial.variables,
                        maximum_terms=_MAX_FACTOR_OUTPUT_TERMS,
                    ),
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

    reconstructed_poly = _result_polynomial(
        reconstructed,
        polynomial.variables,
        maximum_terms=_MAX_FACTOR_OUTPUT_TERMS,
    )

    return MultivariateFactorResult._from_kernel(
        status="FACTORIZED",
        coefficient=coefficient_value,
        factors=factors,
        reconstructed=reconstructed_poly,
    )
