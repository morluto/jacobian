"""Exact action of a univariate shift operator on a proper hypergeometric term."""

from __future__ import annotations

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras._models import ShiftOreOperator
from jacobian.math.ore_algebras.operations import _admit_shift_operator
from jacobian.math.ore_algebras.proper_hypergeometric_actions._models import (
    ProperHypergeometricOperatorActionResult,
)
from jacobian.math.ore_algebras.proper_hypergeometric_terms._models import (
    ProperHypergeometricTerm,
)
from jacobian.math.ore_algebras.proper_hypergeometric_terms.shift_quotients import (
    proper_hypergeometric_n_shift_quotient,
)
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import RationalFunction

_VARIABLES = ("n", "k")
_MAX_ACTION_TERMS = 256
_MAX_ACTION_DEGREE = 128
_MAX_ACTION_DIGITS = 128


def _as_term(value: ProperHypergeometricTerm) -> ProperHypergeometricTerm:
    try:
        if not isinstance(value, ProperHypergeometricTerm):
            raise TypeError("term must be a ProperHypergeometricTerm")
        return ProperHypergeometricTerm.model_validate(value.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("term",),
            code="ore_algebra.proper_hypergeometric_term",
            message="term must be a canonical proper hypergeometric value",
        ) from exc


def _term_count_bound(
    operator: ShiftOreOperator, n_ratio: RationalFunction
) -> tuple[int, int]:
    """Bound supports using the bivariate total-degree monomial envelope."""
    numerator_degree, denominator_degree = _degree_bound(operator, n_ratio)
    numerator_bound = (numerator_degree + 1) * (numerator_degree + 2) // 2
    denominator_bound = (denominator_degree + 1) * (denominator_degree + 2) // 2
    if denominator_bound > _MAX_ACTION_TERMS or numerator_bound > _MAX_ACTION_TERMS:
        raise OperationResourceAdmissionError(
            location=("operator",),
            code="ore_algebra.hypergeometric_action_term_budget",
            message=(
                "the shift action may exceed the 256-term rational-function "
                "expansion budget"
            ),
        )
    return numerator_bound, denominator_bound


def _degree_bound(
    operator: ShiftOreOperator, n_ratio: RationalFunction
) -> tuple[int, int]:
    ratio_num_degree = max(
        (sum(item.exponents) for item in n_ratio.numerator.terms), default=0
    )
    ratio_den_degree = max(
        (sum(item.exponents) for item in n_ratio.denominator.terms), default=0
    )
    numerator_degrees = []
    denominator_degrees = []
    for term in operator.terms:
        coefficient = term.coefficient
        coeff_num = max(
            (item.exponents[0] for item in coefficient.numerator.terms), default=0
        )
        coeff_den = max(
            (item.exponents[0] for item in coefficient.denominator.terms), default=0
        )
        numerator_degrees.append(coeff_num + term.exponent * ratio_num_degree)
        denominator_degrees.append(coeff_den + term.exponent * ratio_den_degree)
    denominator_degree = sum(denominator_degrees)
    numerator_degree = denominator_degree + max(
        (
            numerator_degrees[i] - denominator_degrees[i]
            for i in range(len(numerator_degrees))
        ),
        default=0,
    )
    if max(denominator_degree, numerator_degree) > _MAX_ACTION_DEGREE:
        raise OperationResourceAdmissionError(
            location=("operator",),
            code="ore_algebra.hypergeometric_action_degree_budget",
            message="the shift action may exceed the 128-degree rational-function bound",
        )
    return numerator_degree, denominator_degree


def _coefficient_digits(value: RationalFunction) -> int:
    return max(
        (
            max(
                len(str(abs(term.coefficient.as_integer_ratio()[0]))),
                len(str(term.coefficient.as_integer_ratio()[1])),
            )
            for polynomial in (value.numerator, value.denominator)
            for term in polynomial.terms
        ),
        default=1,
    )


def _digit_bound(operator: ShiftOreOperator, n_ratio: RationalFunction) -> int:
    ratio_digits = _coefficient_digits(n_ratio)
    ratio_num_degree = max(
        (sum(item.exponents) for item in n_ratio.numerator.terms), default=0
    )
    ratio_den_degree = max(
        (sum(item.exponents) for item in n_ratio.denominator.terms), default=0
    )
    ratio_num_terms = len(n_ratio.numerator.terms)
    ratio_den_terms = len(n_ratio.denominator.terms)
    denominator_term_digits = []
    numerator_term_digits = []
    for term in operator.terms:
        exponent = term.exponent
        shifted_num_digits = ratio_digits + ratio_num_degree * (
            len(str(max(1, exponent))) + 1
        ) + len(str(max(1, ratio_num_terms))) + len(
            str(max(1, ratio_num_degree + 1))
        )
        shifted_den_digits = ratio_digits + ratio_den_degree * (
            len(str(max(1, exponent))) + 1
        ) + len(str(max(1, ratio_den_terms))) + len(
            str(max(1, ratio_den_degree + 1))
        )
        coefficient_digits = _coefficient_digits(term.coefficient)
        denominator_term_digits.append(
            coefficient_digits + exponent * shifted_den_digits
        )
        numerator_term_digits.append(coefficient_digits + exponent * shifted_num_digits)
    denominator_digits = sum(denominator_term_digits)
    numerator_digits = max(
        (
            numerator_term_digits[index]
            + denominator_digits
            - denominator_term_digits[index]
            for index, term in enumerate(operator.terms)
        ),
        default=1,
    )
    return max(denominator_digits, numerator_digits)


def _admit_action(operator: ShiftOreOperator, n_ratio: RationalFunction) -> None:
    if not operator.terms:
        return
    numerator_terms, denominator_terms = _term_count_bound(operator, n_ratio)
    coefficient_sum_digits = len(str(max(numerator_terms, denominator_terms, 1)))
    if _digit_bound(operator, n_ratio) + coefficient_sum_digits > _MAX_ACTION_DIGITS:
        raise OperationResourceAdmissionError(
            location=("operator",),
            code="ore_algebra.hypergeometric_action_digit_budget",
            message="the shift action may exceed the 128-digit coefficient bound",
        )


def _apply(
    operator: ShiftOreOperator, term: ProperHypergeometricTerm
) -> RationalFunction:
    from sympy import Integer

    request_checkpoint("before hypergeometric operator action normalization")
    n, k = symbols_for_variables(_VARIABLES)
    n_quotient = proper_hypergeometric_n_shift_quotient(term)
    _admit_action(operator, n_quotient)
    n_ratio = rational_function_to_sympy(n_quotient, symbols=(n, k))
    total = Integer(0)
    for item in operator.terms:
        coefficient = rational_function_to_sympy(item.coefficient, symbols=(n,))
        shifted_product = Integer(1)
        for amount in range(item.exponent):
            request_checkpoint("during hypergeometric operator action substitution")
            shifted_product *= n_ratio.subs(n, n + amount)
        total += coefficient * shifted_product
    request_checkpoint("before hypergeometric operator action normalization")
    return rational_function_from_sympy(
        total,
        _VARIABLES,
        maximum_terms=_MAX_ACTION_TERMS,
        deadline_check=lambda: request_checkpoint(
            "during hypergeometric operator action normalization"
        ),
        symbols=(n, k),
    )


def proper_hypergeometric_operator_action(
    operator: ShiftOreOperator,
    term: ProperHypergeometricTerm,
) -> ProperHypergeometricOperatorActionResult:
    """Return R(n,k) with ``P*T = R*T`` on the common generic locus."""
    term = _as_term(term)
    operator = _admit_shift_operator(operator, label="operator")
    if term.is_zero or not operator.terms:
        # The zero summand and zero operator use the same canonical relative
        # multiplier, without claiming a pointwise quotient for the zero term.
        from jacobian.math.ore_algebras.proper_hypergeometric_actions._models import (
            _zero_rational_function,
        )

        relative_multiplier = _zero_rational_function()
    else:
        relative_multiplier = _apply(operator, term)
    return ProperHypergeometricOperatorActionResult(
        operator=operator,
        term=term,
        relative_multiplier=relative_multiplier,
    )


__all__ = ["proper_hypergeometric_operator_action"]
