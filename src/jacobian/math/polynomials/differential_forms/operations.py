"""Exact sparse wedge products of polynomial differential forms."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import (
    MAX_CANONICAL_INTEGER_DIGITS,
    CanonicalRational,
)
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.differential_forms.values import (
    MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS,
    MAX_DIFFERENTIAL_FORM_COMPONENTS,
    MAX_DIFFERENTIAL_FORM_EXPONENT,
    MAX_DIFFERENTIAL_FORM_TERMS,
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


_MergedPair = tuple[FormComponent, FormComponent, tuple[int, ...], int]
_RemainingTerms = dict[tuple[int, ...], dict[tuple[int, ...], Fraction]]


def _fraction_component_digits(value: Fraction) -> int:
    return max(
        len(format_canonical_integer(abs(value.numerator))),
        len(format_canonical_integer(value.denominator)),
    )


def _convolve_pairs(pairs: tuple[_MergedPair, ...]) -> _RemainingTerms:
    aggregate: _RemainingTerms = {}
    for first, second, indices, sign in pairs:
        terms = aggregate.setdefault(indices, {})
        for exponents, coefficient in _multiply_components(
            first.coefficient, second.coefficient, sign
        ).items():
            terms[exponents] = terms.get(exponents, Fraction()) + coefficient
    return {
        indices: {
            exponents: coefficient
            for exponents, coefficient in terms.items()
            if coefficient
        }
        for indices, terms in aggregate.items()
    }


def _admit_remaining_support(aggregate: _RemainingTerms) -> None:
    """Cap surviving monomials after signed convolution cancellation."""

    if any(
        len(terms) > MAX_DIFFERENTIAL_FORM_TERMS for terms in aggregate.values()
    ):
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="differential_form.wedge.output_budget",
            message="wedge coefficient support exceeds the bounded output envelope",
        )


def _admit_remaining_coefficients(aggregate: _RemainingTerms) -> None:
    """Cap surviving cancelled rationals after signed convolution."""

    bounds = tuple(
        _fraction_component_digits(coefficient)
        for terms in aggregate.values()
        for coefficient in terms.values()
    )
    if (
        any(height > MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS for height in bounds)
        or sum(height * height for height in bounds) > 100_000_000
    ):
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="differential_form.wedge.coefficient_budget",
            message="wedge coefficient growth exceeds the admitted digit-work or output envelope",
        )


def _admit_degree(degree: int) -> None:
    """Keep an overflowing canonical zero degree a typed resource rejection."""

    if degree.bit_length() <= 3 * MAX_CANONICAL_INTEGER_DIGITS:
        return
    if degree >= 10**MAX_CANONICAL_INTEGER_DIGITS:
        raise OperationResourceAdmissionError(
            location=("left", "degree"),
            code="differential_form.wedge.degree_budget",
            message="wedge degree exceeds the canonical integer representation envelope",
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
    _admit_degree(degree)
    if degree > len(left.variables):
        return PolynomialDifferentialForm(
            variables=left.variables, degree=degree, components=()
        )
    if int(left.degree) % 2 == 1 and left == right:
        # Graded commutativity forces an odd form's self-wedge to vanish
        # before exponent or coefficient expansion.
        return PolynomialDifferentialForm(
            variables=left.variables, degree=degree, components=()
        )
    pairs = tuple(
        (first, second, indices, sign)
        for first in left.components
        for second in right.components
        if (merged := _merged_indices(first.indices, second.indices)) is not None
        for indices, sign in (merged,)
    )
    pair_count = len(pairs)
    if pair_count > MAX_DIFFERENTIAL_FORM_COMPONENTS * MAX_DIFFERENTIAL_FORM_COMPONENTS:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="differential_form.wedge.pair_budget",
            message="wedge component-pair count exceeds the bounded work envelope",
        )
    term_pair_count = sum(
        len(first.coefficient.polynomial.terms)
        * len(second.coefficient.polynomial.terms)
        for first, second, _, _ in pairs
    )
    if term_pair_count > 1_000_000:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="differential_form.wedge.term_budget",
            message="wedge polynomial convolution exceeds the bounded work envelope",
        )
    aggregate = _convolve_pairs(pairs)
    _admit_remaining_support(aggregate)
    _admit_remaining_coefficients(aggregate)
    maximum_exponent = max(
        (
            max(exponents, default=0)
            for terms in aggregate.values()
            for exponents in terms
        ),
        default=0,
    )
    if maximum_exponent > MAX_DIFFERENTIAL_FORM_EXPONENT:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="differential_form.wedge.exponent_budget",
            message="wedge coefficient exponents exceed the bounded output envelope",
        )
    components: list[FormComponent] = []
    for indices in sorted(aggregate):
        terms = aggregate[indices]
        if not terms:
            continue
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
