"""Exact formal shift quotients for proper hypergeometric terms."""

from __future__ import annotations

from math import comb
from typing import Any

from jacobian._execution import request_checkpoint
from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras.proper_hypergeometric_terms._models import (
    ProperHypergeometricTerm,
)
from jacobian.math.ore_algebras.proper_hypergeometric_terms.shift_quotients_models import (
    ProperHypergeometricShiftQuotientsResult,
)
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import RationalFunction

_VARIABLES = ("n", "k")
_MAX_PREFAC_DEGREE = 16
_MAX_PREFAC_TERMS = 32
_MAX_FACTORIAL_RATIO_DEGREE = 12
_MAX_EXPANSION_TERMS = 256
_MAX_OUTPUT_DIGITS = 128


def _term_degree(term: ProperHypergeometricTerm) -> int:
    return max(
        (sum(monomial.exponents) for monomial in term.polynomial.polynomial.terms),
        default=0,
    )


def _shifted_polynomial_term_bound(term: ProperHypergeometricTerm, axis: int) -> int:
    return sum(
        (monomial.exponents[axis] + 1) * (monomial.exponents[1 - axis] + 1)
        for monomial in term.polynomial.polynomial.terms
    )


def _factorial_ratio_degree(term: ProperHypergeometricTerm, axis: int) -> int:
    return sum(
        abs((factor.n_coefficient, factor.k_coefficient)[axis]) * abs(factor.power)
        for factor in term.factorial_factors
    )


def _admit_quotient(term: ProperHypergeometricTerm, axis: int) -> None:
    """Bound rational expansion before constructing backend polynomials."""
    if term.is_zero:
        raise OperationDomainValidationError(
            location=("term",),
            code="ore_algebra.hypergeometric_zero_term",
            message="shift quotients are undefined for the zero term",
        )
    polynomial = term.polynomial.polynomial
    if (
        len(polynomial.terms) > _MAX_PREFAC_TERMS
        or _term_degree(term) > _MAX_PREFAC_DEGREE
    ):
        raise OperationResourceAdmissionError(
            location=("term", "polynomial"),
            code="ore_algebra.hypergeometric_prefactor_budget",
            message="the polynomial prefactor exceeds the shift-quotient envelope",
        )
    factorial_degree = _factorial_ratio_degree(term, axis)
    offset_digits = max(
        (
            decimal_digit_width(abs(factor.offset))
            for factor in term.factorial_factors
            if (factor.n_coefficient, factor.k_coefficient)[axis] != 0
        ),
        default=1,
    )
    if factorial_degree > _MAX_FACTORIAL_RATIO_DEGREE:
        raise OperationResourceAdmissionError(
            location=("term", "factorial_factors"),
            code="ore_algebra.hypergeometric_factorial_ratio_budget",
            message="the factorial shift quotient exceeds the admitted product degree",
        )
    shifted_terms = _shifted_polynomial_term_bound(term, axis)
    factorial_terms = comb(factorial_degree + 2, 2)
    if (
        shifted_terms * factorial_terms > _MAX_EXPANSION_TERMS
        or len(polynomial.terms) * factorial_terms > _MAX_EXPANSION_TERMS
    ):
        raise OperationResourceAdmissionError(
            location=("term",),
            code="ore_algebra.hypergeometric_quotient_expansion_budget",
            message="the exact quotient numerator or denominator may exceed 256 terms",
        )
    coefficient_digits = max(
        (
            max(
                len(str(abs(numerator))),
                len(str(denominator)),
            )
            for monomial in polynomial.terms
            for numerator, denominator in (monomial.coefficient.as_integer_ratio(),)
        ),
        default=1,
    )
    base = term.n_base if axis == 0 else term.k_base
    base_numerator, base_denominator = base.as_integer_ratio()
    base_digits = max(len(str(abs(base_numerator))), len(str(base_denominator)))
    conservative_digits = (
        2 * coefficient_digits
        + 2 * base_digits
        + factorial_degree * (offset_digits + 4)
        + _term_degree(term)
        + len(str(max(1, shifted_terms * factorial_terms)))
    )
    if conservative_digits > _MAX_OUTPUT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("term",),
            code="ore_algebra.hypergeometric_quotient_coefficient_budget",
            message="the exact quotient coefficient bound exceeds 128 digits",
        )


def _factorial_ratio_expression(
    term: ProperHypergeometricTerm, axis: int, variables: tuple[Any, ...]
) -> tuple[Any, Any]:
    from sympy import Integer

    result_num = Integer(1)
    result_den = Integer(1)
    for factor in term.factorial_factors:
        coefficient = (factor.n_coefficient, factor.k_coefficient)[axis]
        if coefficient == 0:
            continue
        affine = (
            factor.n_coefficient * variables[0]
            + factor.k_coefficient * variables[1]
            + factor.offset
        )
        if coefficient > 0:
            ratio_num = Integer(1)
            ratio_den = Integer(1)
            for offset in range(1, coefficient + 1):
                ratio_num *= affine + offset
        else:
            ratio_num = Integer(1)
            ratio_den = Integer(1)
            for offset in range(-coefficient):
                ratio_den *= affine - offset
        if factor.power > 0:
            result_num *= ratio_num**factor.power
            result_den *= ratio_den**factor.power
        else:
            result_num *= ratio_den ** (-factor.power)
            result_den *= ratio_num ** (-factor.power)
    return result_num, result_den


def _quotient(term: ProperHypergeometricTerm, axis: int) -> RationalFunction:
    from sympy import Rational

    request_checkpoint("before hypergeometric shift quotient normalization")
    variables = symbols_for_variables(_VARIABLES)
    polynomial = rational_polynomial_to_sympy(term.polynomial)
    shift = variables[axis]
    shifted = polynomial.subs(shift, shift + 1)
    factorial_num, factorial_den = _factorial_ratio_expression(term, axis, variables)
    base = term.n_base if axis == 0 else term.k_base
    base_expression = Rational(*base.as_integer_ratio())
    expression = (
        base_expression * shifted * factorial_num / (polynomial * factorial_den)
    )
    result = rational_function_from_sympy(
        expression,
        _VARIABLES,
        maximum_terms=_MAX_EXPANSION_TERMS,
        deadline_check=lambda: request_checkpoint(
            "during hypergeometric shift quotient normalization"
        ),
        symbols=variables,
    )
    request_checkpoint("after hypergeometric shift quotient normalization")
    if any(
        max(
            len(str(abs(coefficient.as_fraction().numerator))),
            len(str(coefficient.as_fraction().denominator)),
        )
        > _MAX_OUTPUT_DIGITS
        for polynomial_part in (result.numerator, result.denominator)
        for coefficient in (item.coefficient for item in polynomial_part.terms)
    ):
        raise OperationResourceAdmissionError(
            location=("result",),
            code="ore_algebra.hypergeometric_quotient_output_budget",
            message="the reduced shift quotient exceeds the 128-digit output envelope",
        )
    return result


def proper_hypergeometric_shift_quotients(
    term: ProperHypergeometricTerm,
) -> ProperHypergeometricShiftQuotientsResult:
    """Return the formal rational quotients ``T(n+1,k)/T(n,k)`` and
    ``T(n,k+1)/T(n,k)``.

    The equalities are in ``QQ(n,k)`` and describe the generic nonzero-term
    locus. They do not define pointwise quotients at zeros, poles, or the
    reciprocal-factorial support boundary.
    """
    term = ProperHypergeometricTerm.model_validate(term.model_dump())
    _admit_quotient(term, 0)
    _admit_quotient(term, 1)
    return ProperHypergeometricShiftQuotientsResult(
        n_ratio=_quotient(term, 0), k_ratio=_quotient(term, 1)
    )


def proper_hypergeometric_n_shift_quotient(
    term: ProperHypergeometricTerm,
) -> RationalFunction:
    """Compute only ``T(n+1,k)/T(n,k)`` for internal n-shift consumers."""
    term = ProperHypergeometricTerm.model_validate(term.model_dump())
    _admit_quotient(term, 0)
    return _quotient(term, 0)
