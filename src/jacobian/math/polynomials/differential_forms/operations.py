"""Exact sparse wedge products of polynomial differential forms."""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from pydantic import ValidationError

from jacobian._exact import (
    MAX_CANONICAL_INTEGER_DIGITS,
    CanonicalRational,
)
from jacobian._execution import execution_deadline, request_checkpoint
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

_MergedPair = tuple[FormComponent, FormComponent, tuple[int, ...], int]
_RemainingTerms = dict[tuple[int, ...], dict[tuple[int, ...], Fraction]]
_CONVOLUTION_CHECKPOINT_INTERVAL = 256
WEDGE_WALL_SECONDS = 60.0


def _integer_decimal_digits(value: int) -> int:
    magnitude = abs(value)
    if magnitude < 10:
        return 1
    return (magnitude.bit_length() * 30103) // 100000 + 1


def _fraction_component_digits(value: Fraction) -> int:
    return max(
        len(format_canonical_integer(abs(value.numerator))),
        len(format_canonical_integer(value.denominator)),
    )


def _unit_coefficient(value: CanonicalRational) -> bool:
    return value.den == 1 and value.num in (-1, 1)


def _cancelled_product_digit_bound(
    left: CanonicalRational, right: CanonicalRational
) -> int:
    """Upper-bound digits of ``left*right`` after num/den cross-cancellation."""

    if _unit_coefficient(left):
        return max(
            len(format_canonical_integer(abs(right.num))),
            len(format_canonical_integer(right.den)),
        )
    if _unit_coefficient(right):
        return max(
            len(format_canonical_integer(abs(left.num))),
            len(format_canonical_integer(left.den)),
        )
    left_num, left_den = abs(left.num), left.den
    right_num, right_den = abs(right.num), right.den
    cross_left = gcd(left_num, right_den)
    cross_right = gcd(right_num, left_den)
    return max(
        len(format_canonical_integer(left_num // cross_left))
        + len(format_canonical_integer(right_num // cross_right)),
        len(format_canonical_integer(left_den // cross_right))
        + len(format_canonical_integer(right_den // cross_left)),
    )


def _bounded_fraction_add(current: Fraction, value: Fraction) -> Fraction:
    """Add exact rationals, refusing unadmitted common-denominator growth first.

    Partial sums may briefly exceed the output digit cap; the surviving
    monomial height is enforced after all signed contributions are included.
    """

    if not current:
        return value
    if not value:
        return current
    left_den, right_den = current.denominator, value.denominator
    if left_den != right_den:
        overlap = gcd(left_den, right_den)
        den_digits = (
            _integer_decimal_digits(left_den)
            + _integer_decimal_digits(right_den)
            - _integer_decimal_digits(overlap)
        )
        if den_digits > MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS:
            _coefficient_budget()
    return current + value


def _coefficient_budget() -> None:
    raise OperationResourceAdmissionError(
        location=("left", "right"),
        code="differential_form.wedge.coefficient_budget",
        message="wedge coefficient growth exceeds the admitted digit-work or output envelope",
    )


def _admit_cancelled_product_heights(pairs: tuple[_MergedPair, ...]) -> None:
    """Reject oversized rational products before convolution materializes them."""

    completed = 0
    for first, second, _, _ in pairs:
        for left_term in first.coefficient.polynomial.terms:
            for right_term in second.coefficient.polynomial.terms:
                completed += 1
                if completed % _CONVOLUTION_CHECKPOINT_INTERVAL == 0:
                    request_checkpoint(
                        "during differential wedge product-height admission"
                    )
                bound = _cancelled_product_digit_bound(
                    left_term.coefficient, right_term.coefficient
                )
                if bound <= MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS:
                    continue
                product = (
                    left_term.coefficient.as_fraction()
                    * right_term.coefficient.as_fraction()
                )
                if (
                    _fraction_component_digits(product)
                    > MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS
                ):
                    _coefficient_budget()


def _merged_indices(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[tuple[int, ...], int] | None:
    if set(left).intersection(right):
        return None
    inversions = sum(index > other for index in left for other in right)
    return tuple(sorted((*left, *right))), -1 if inversions % 2 else 1


def _convolve_pairs(pairs: tuple[_MergedPair, ...]) -> _RemainingTerms:
    aggregate: _RemainingTerms = {}
    completed = 0
    for first, second, indices, sign in pairs:
        terms = aggregate.setdefault(indices, {})
        for left_term in first.coefficient.polynomial.terms:
            for right_term in second.coefficient.polynomial.terms:
                completed += 1
                if completed % _CONVOLUTION_CHECKPOINT_INTERVAL == 0:
                    request_checkpoint("during differential wedge convolution")
                exponents = tuple(
                    a + b
                    for a, b in zip(
                        left_term.exponents, right_term.exponents, strict=True
                    )
                )
                value = (
                    sign
                    * left_term.coefficient.as_fraction()
                    * right_term.coefficient.as_fraction()
                )
                combined = _bounded_fraction_add(
                    terms.get(exponents, Fraction()), value
                )
                terms[exponents] = combined
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

    if any(len(terms) > MAX_DIFFERENTIAL_FORM_TERMS for terms in aggregate.values()):
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="differential_form.wedge.output_budget",
            message="wedge coefficient support exceeds the bounded output envelope",
        )


def _is_scalar_unit(form: PolynomialDifferentialForm) -> bool:
    if int(form.degree) != 0 or len(form.components) != 1:
        return False
    component = form.components[0]
    if component.indices:
        return False
    terms = component.coefficient.polynomial.terms
    if len(terms) != 1:
        return False
    term = terms[0]
    return _unit_coefficient(term.coefficient) and all(
        exponent == 0 for exponent in term.exponents
    )


def _admit_remaining_coefficients(aggregate: _RemainingTerms) -> None:
    """Cap surviving cancelled rationals after signed convolution."""

    bounds = tuple(
        _fraction_component_digits(coefficient)
        for terms in aggregate.values()
        for coefficient in terms.values()
    )
    if any(height > MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS for height in bounds):
        _coefficient_budget()


def _admit_form(
    value: object, *, location: tuple[str, ...]
) -> PolynomialDifferentialForm:
    """Revalidate a native operand before reading fields or expanding products."""

    if not isinstance(value, PolynomialDifferentialForm):
        raise OperationDomainValidationError(
            location=location,
            code="differential_form.operand_type",
            message="wedge operands must be polynomial differential forms",
        )
    degree = getattr(value, "degree", None)
    if not isinstance(degree, int) or degree < 0:
        raise OperationDomainValidationError(
            location=(*location, "degree"),
            code="differential_form.degree",
            message="wedge operands must have a nonnegative degree",
        )
    try:
        return PolynomialDifferentialForm.model_validate(
            value.model_dump(warnings="none")
        )
    except ValidationError as error:
        detail = error.errors()[0]
        raise OperationDomainValidationError(
            location=(*location, *tuple(detail.get("loc", ()))),
            code=str(detail["type"]),
            message=str(detail["msg"]),
        ) from error


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

    execution_deadline(WEDGE_WALL_SECONDS)
    left = _admit_form(left, location=("left",))
    right = _admit_form(right, location=("right",))
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
    if _is_scalar_unit(right):
        return left
    if _is_scalar_unit(left):
        return right
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
    _admit_cancelled_product_heights(pairs)
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
