"""Exact sparse wedge products of polynomial differential forms."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.differential_forms.values import (
    MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS,
    MAX_DIFFERENTIAL_FORM_COMPONENTS,
    MAX_DIFFERENTIAL_FORM_EXPONENT,
    FormComponent,
    PolynomialDifferentialForm,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _multiply_components(
    left: RationalPolynomial, right: RationalPolynomial, sign: int
) -> dict[tuple[int, ...], Fraction]:
    values: dict[tuple[int, ...], Fraction] = {}
    for first in left.polynomial.terms:
        for second in right.polynomial.terms:
            exponents = tuple(
                a + b for a, b in zip(first.exponents, second.exponents, strict=True)
            )
            value = (
                sign
                * first.coefficient.as_fraction()
                * second.coefficient.as_fraction()
            )
            values[exponents] = values.get(exponents, Fraction()) + value
    return {exponents: value for exponents, value in values.items() if value}


def _merged_indices(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[tuple[int, ...], int] | None:
    if set(left).intersection(right):
        return None
    inversions = sum(index > other for index in left for other in right)
    return tuple(sorted((*left, *right))), -1 if inversions % 2 else 1


def _admit_coefficient_growth(
    left: PolynomialDifferentialForm, right: PolynomialDifferentialForm
) -> None:
    # Bound rational accumulation independently for each differential basis
    # before multiplying coefficients. Summing N fractions of height h costs
    # at most N * (h + 1) decimal digits with an unreduced product denominator.
    projected_digits: dict[tuple[int, ...], int] = {}
    for first in left.components:
        for second in right.components:
            merged = _merged_indices(first.indices, second.indices)
            if merged is None:
                continue
            indices, _ = merged
            first_terms = first.coefficient.polynomial.terms
            second_terms = second.coefficient.polynomial.terms
            products = len(first_terms) * len(second_terms)
            height = (
                max(
                    canonical_rational_component_digits(term.coefficient)
                    for term in first_terms
                )
                + max(
                    canonical_rational_component_digits(term.coefficient)
                    for term in second_terms
                )
                + 1
            )
            projected_digits[indices] = (
                projected_digits.get(indices, 0) + products * height
            )
    if (
        any(
            height > MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS
            for height in projected_digits.values()
        )
        or sum(height * height for height in projected_digits.values()) > 100_000_000
    ):
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="differential_form.wedge.coefficient_budget",
            message="wedge coefficient growth exceeds the admitted digit-work or output envelope",
        )


def wedge(
    left: PolynomialDifferentialForm, right: PolynomialDifferentialForm
) -> PolynomialDifferentialForm:
    """Return the exact graded-commutative wedge product."""

    if left.variables != right.variables:
        raise OperationDomainValidationError(
            location=("right", "variables"),
            code="differential_form.variable_axis",
            message="forms must use one identical ordered variable axis",
        )
    degree = left.degree + right.degree
    if degree > len(left.variables):
        return PolynomialDifferentialForm(
            variables=left.variables, degree=degree, components=()
        )
    pair_count = len(left.components) * len(right.components)
    if pair_count > MAX_DIFFERENTIAL_FORM_COMPONENTS * MAX_DIFFERENTIAL_FORM_COMPONENTS:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="differential_form.wedge.pair_budget",
            message="wedge component-pair count exceeds the bounded work envelope",
        )
    term_pair_count = sum(
        len(first.coefficient.polynomial.terms)
        * len(second.coefficient.polynomial.terms)
        for first in left.components
        for second in right.components
    )
    if term_pair_count > 1_000_000:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="differential_form.wedge.term_budget",
            message="wedge polynomial convolution exceeds the bounded work envelope",
        )
    _admit_coefficient_growth(left, right)
    maximum_exponent = max(
        (
            first_term.exponents[axis] + second_term.exponents[axis]
            for first in left.components
            for second in right.components
            for first_term in first.coefficient.polynomial.terms
            for second_term in second.coefficient.polynomial.terms
            for axis in range(len(left.variables))
        ),
        default=0,
    )
    if maximum_exponent > MAX_DIFFERENTIAL_FORM_EXPONENT:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="differential_form.wedge.exponent_budget",
            message="wedge coefficient exponents exceed the bounded output envelope",
        )
    aggregate: dict[tuple[int, ...], dict[tuple[int, ...], Fraction]] = {}
    for first in left.components:
        for second in right.components:
            merged = _merged_indices(first.indices, second.indices)
            if merged is None:
                continue
            indices, sign = merged
            terms = aggregate.setdefault(indices, {})
            for exponents, coefficient in _multiply_components(
                first.coefficient, second.coefficient, sign
            ).items():
                terms[exponents] = terms.get(exponents, Fraction()) + coefficient
    components: list[FormComponent] = []
    for indices in sorted(aggregate):
        terms = {
            exponents: coefficient
            for exponents, coefficient in aggregate[indices].items()
            if coefficient
        }
        if not terms:
            continue
        if len(terms) > 256:
            raise OperationResourceAdmissionError(
                location=("left", "right"),
                code="differential_form.wedge.output_budget",
                message="wedge coefficient support exceeds the bounded output envelope",
            )
        polynomial_coefficient = RationalPolynomial(
            variables=left.variables,
            polynomial=SparseRationalPolynomial(
                terms=tuple(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational.from_fraction(value),
                        exponents=exponents,
                    )
                    for exponents, value in sorted(terms.items(), reverse=True)
                )
            ),
        )
        components.append(
            FormComponent(indices=indices, coefficient=polynomial_coefficient)
        )
    return PolynomialDifferentialForm(
        variables=left.variables, degree=degree, components=tuple(components)
    )


__all__ = ["wedge"]
