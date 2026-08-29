"""Exact finite kernels for the reviewed level-one named forms.

This module intentionally contains no wire models.  It is the shared source
of the coefficient formulas, result-envelope proof, and reconstruction used
by the native constructor and its canonical value validation.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import Literal

from jacobian.catalog.models import OperationDomainValidationError

NamedLevelOneModularForm = Literal["E4", "E6", "DELTA"]

NAMED_LEVEL_ONE_FORMS = frozenset(("E4", "E6", "DELTA"))

# This ceiling charges all coefficient products in three schoolbook truncated
# products. FLINT selects faster kernels where applicable, so the estimate is
# conservative without depending on one private multiplication algorithm.
MAX_LEVEL_ONE_WORK_TERMS = 4_000_000
MAX_LEVEL_ONE_SERIALIZED_CHARACTERS = 65_536


def divisor_power_sum(index: int, exponent: int) -> int:
    """Return ``sigma_exponent(index)`` by its complete divisor-pair scan."""
    if index < 1:
        raise ValueError(
            "divisor-power sums are defined here only for positive indices"
        )
    total = 0
    for divisor in range(1, isqrt(index) + 1):
        if index % divisor:
            continue
        quotient = index // divisor
        total += divisor**exponent
        if quotient != divisor:
            total += quotient**exponent
    return total


def eisenstein_coefficients(
    form: Literal["E4", "E6"], truncation_order: int
) -> tuple[Fraction, ...]:
    """Return the canonical q-prefix of normalized E4 or E6."""
    if form == "E4":
        factor, exponent = 240, 3
    else:
        factor, exponent = -504, 5
    return (
        Fraction(1),
        *(
            Fraction(factor * divisor_power_sum(index, exponent))
            for index in range(1, truncation_order)
        ),
    )


def expected_coefficients(
    form: NamedLevelOneModularForm, truncation_order: int
) -> tuple[Fraction, ...]:
    """Return the unique normalized prefix for one closed named form."""
    if form == "E4":
        return eisenstein_coefficients("E4", truncation_order)
    if form == "E6":
        return eisenstein_coefficients("E6", truncation_order)
    e4 = eisenstein_coefficients("E4", truncation_order)
    e6 = eisenstein_coefficients("E6", truncation_order)
    from jacobian.math.number_theory.modular_forms._flint import delta_coefficients

    return delta_coefficients(e4, e6)


def metadata(
    form: NamedLevelOneModularForm,
) -> tuple[int, Literal["HOLOMORPHIC", "CUSP"], str]:
    """Return the exact level-one parent and normalization for ``form``."""
    if form == "E4":
        return 4, "HOLOMORPHIC", "CONSTANT_TERM_ONE__E4_COEFFICIENT_240_SIGMA_3"
    if form == "E6":
        return 6, "HOLOMORPHIC", "CONSTANT_TERM_ONE__E6_COEFFICIENT_MINUS_504_SIGMA_5"
    return 12, "CUSP", "DELTA_EQUALS_E4_CUBED_MINUS_E6_SQUARED_OVER_1728"


def coefficient_digit_bound(
    form: NamedLevelOneModularForm, truncation_order: int
) -> int:
    """Conservative decimal bound for every integral output coefficient.

    For n < P, ``sigma_r(n) <= n^(r+1) <= P^(r+1)``. For Delta, Deligne's
    bound ``|tau(n)| <= d(n)n^(11/2)`` and ``d(n) <= 2sqrt(n)`` give the
    integral bound ``|tau(n)| <= 2P^6``. Admission therefore tracks the
    canonical result rather than unreduced intermediates in its identity.
    """
    p = truncation_order
    e4_bound = 240 * p**4
    e6_bound = 504 * p**6
    if form == "E4":
        bound = e4_bound
    elif form == "E6":
        bound = e6_bound
    else:
        bound = 2 * p**6
    return len(str(bound))


def require_level_one_admission(
    form: NamedLevelOneModularForm, truncation_order: int
) -> None:
    """Prove finite scan, series work, coefficient, and output envelopes."""
    if form not in NAMED_LEVEL_ONE_FORMS:
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.form_out_of_range",
            message="form must be one of 'E4', 'E6', or 'DELTA'",
        )
    if isinstance(truncation_order, bool) or not isinstance(truncation_order, int):
        raise OperationDomainValidationError(
            location=("truncation_order",),
            code="modular_form.truncation_order_must_be_integer",
            message="truncation_order must be a plain integer",
        )
    if truncation_order < 1:
        raise OperationDomainValidationError(
            location=("truncation_order",),
            code="modular_form.truncation_order_out_of_range",
            message="truncation_order must be positive",
        )

    p = truncation_order
    divisor_scans = p * isqrt(p)
    formula_scans = divisor_scans if form in {"E4", "E6"} else 2 * divisor_scans
    series_terms = 0 if form in {"E4", "E6"} else 3 * p * (p + 1) // 2
    if formula_scans + series_terms > MAX_LEVEL_ONE_WORK_TERMS:
        raise OperationDomainValidationError(
            location=("truncation_order",),
            code="modular_form.exact_work_bound_exceeded",
            message="level-one q-expansion exceeds the exact work bound",
        )

    digits = coefficient_digit_bound(form, p)
    # ``num`` and ``den`` are canonical integer strings; these fixed 28
    # characters cover their JSON punctuation, field names, and a separator.
    serialized_characters = 512 + p * (digits + 28)
    if serialized_characters > MAX_LEVEL_ONE_SERIALIZED_CHARACTERS:
        raise OperationDomainValidationError(
            location=("truncation_order",),
            code="modular_form.serialized_result_bound_exceeded",
            message="level-one q-expansion exceeds the serialized result bound",
        )


__all__ = [
    "MAX_LEVEL_ONE_SERIALIZED_CHARACTERS",
    "MAX_LEVEL_ONE_WORK_TERMS",
    "NAMED_LEVEL_ONE_FORMS",
    "NamedLevelOneModularForm",
    "coefficient_digit_bound",
    "divisor_power_sum",
    "eisenstein_coefficients",
    "expected_coefficients",
    "metadata",
    "require_level_one_admission",
]
