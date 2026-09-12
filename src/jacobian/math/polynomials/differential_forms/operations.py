"""Exact sparse wedge products of polynomial differential forms."""

from __future__ import annotations

from fractions import Fraction
from math import gcd

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
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

_MergedPair = tuple[FormComponent, FormComponent, tuple[int, ...], int]
_ContributionMap = dict[tuple[int, ...], dict[tuple[int, ...], list[Fraction]]]
_RemainingTerms = dict[tuple[int, ...], dict[tuple[int, ...], Fraction]]
_CONVOLUTION_CHECKPOINT_INTERVAL = 256
WEDGE_WALL_SECONDS = 60.0
MAX_WEDGE_TERM_PAIRS = 1_000_000
# Schoolbook rational multiplication is quadratic in component height.  Bind
# pair counts to those heights before expanding products; wall time is not
# a work bound.
MAX_WEDGE_DIGIT_WORK = 100_000_000
# Keep the complete serialized result inside the 10 MiB canonical output
# envelope rather than only per-component term and digit caps.
MAX_WEDGE_OUTPUT_DIGITS = 4_000_000


def _integer_decimal_digits(value: int) -> int:
    magnitude = abs(value)
    if magnitude < 10:
        return 1
    return (magnitude.bit_length() * 30103) // 100000 + 1


def _rational_height_digits(value: CanonicalRational) -> int:
    numerator, denominator = value.as_integer_ratio()
    return max(
        _integer_decimal_digits(abs(numerator)),
        _integer_decimal_digits(denominator),
    )


def _fraction_component_digits(value: Fraction) -> int:
    return max(
        len(format_canonical_integer(abs(value.numerator))),
        len(format_canonical_integer(value.denominator)),
    )


def _fraction_serialized_digits(value: Fraction) -> int:
    return len(format_canonical_integer(abs(value.numerator))) + len(
        format_canonical_integer(value.denominator)
    )


def _unit_coefficient(value: CanonicalRational) -> bool:
    numerator, denominator = value.as_integer_ratio()
    return denominator == 1 and numerator in (-1, 1)


def _bounded_fraction_add(current: Fraction, value: Fraction) -> Fraction:
    """Add exact rationals, refusing unadmitted common-denominator growth first.

    Same-denominator contributions are cancelled before this helper runs.
    Distinct remaining denominators still cannot grow past the output envelope.
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


def _sum_signed_fractions(values: list[Fraction]) -> Fraction:
    """Cancel same-denominator signed contributions before any LCM or height cap."""

    by_denominator: dict[int, int] = {}
    for value in values:
        if not value:
            continue
        denominator = value.denominator
        by_denominator[denominator] = (
            by_denominator.get(denominator, 0) + value.numerator
        )
    remaining: list[Fraction] = []
    for denominator, numerator in by_denominator.items():
        if numerator:
            remaining.append(Fraction(numerator, denominator))
    total = Fraction()
    for value in remaining:
        total = _bounded_fraction_add(total, value)
    return total


def _coefficient_budget() -> None:
    raise OperationResourceAdmissionError(
        location=("left", "right"),
        code="differential_form.wedge.coefficient_budget",
        message="wedge coefficient growth exceeds the admitted digit-work or output envelope",
    )


def _output_budget(message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("left", "right"),
        code="differential_form.wedge.output_budget",
        message=message,
    )


def _term_budget() -> None:
    raise OperationResourceAdmissionError(
        location=("left", "right"),
        code="differential_form.wedge.term_budget",
        message="wedge polynomial convolution exceeds the bounded work envelope",
    )


def _zero_form(
    variables: tuple[PolynomialVariable, ...], degree: int
) -> PolynomialDifferentialForm:
    return PolynomialDifferentialForm._from_admitted(
        variables=variables, degree=degree, components=()
    )


def _merged_indices(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[tuple[int, ...], int] | None:
    if set(left).intersection(right):
        return None
    inversions = sum(index > other for index in left for other in right)
    return tuple(sorted((*left, *right))), -1 if inversions % 2 else 1


def _collect_contributions(pairs: tuple[_MergedPair, ...]) -> _ContributionMap:
    grouped: _ContributionMap = {}
    completed = 0
    digit_work = 0
    for first, second, indices, sign in pairs:
        terms = grouped.setdefault(indices, {})
        for left_term in first.coefficient.polynomial.terms:
            left_digits = _rational_height_digits(left_term.coefficient)
            for right_term in second.coefficient.polynomial.terms:
                completed += 1
                if completed % _CONVOLUTION_CHECKPOINT_INTERVAL == 0:
                    request_checkpoint("during differential wedge convolution")
                right_digits = _rational_height_digits(right_term.coefficient)
                digit_work += left_digits * right_digits
                if digit_work > MAX_WEDGE_DIGIT_WORK:
                    _term_budget()
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
                terms.setdefault(exponents, []).append(value)
    return grouped


def _reduce_contributions(grouped: _ContributionMap) -> _RemainingTerms:
    """Sum signed products per monomial, then cap surviving height and size."""

    aggregate: _RemainingTerms = {}
    completed = 0
    output_digits = 0
    for indices, terms in grouped.items():
        reduced: dict[tuple[int, ...], Fraction] = {}
        for exponents, contributions in terms.items():
            completed += 1
            if completed % _CONVOLUTION_CHECKPOINT_INTERVAL == 0:
                request_checkpoint("during differential wedge cancellation")
            coefficient = _sum_signed_fractions(contributions)
            if not coefficient:
                continue
            if (
                _fraction_component_digits(coefficient)
                > MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS
            ):
                _coefficient_budget()
            output_digits += _fraction_serialized_digits(coefficient)
            if output_digits > MAX_WEDGE_OUTPUT_DIGITS:
                _output_budget(
                    "wedge serialized coefficient digits exceed the bounded output envelope"
                )
            reduced[exponents] = coefficient
        if reduced:
            aggregate[indices] = reduced
    return aggregate


def _admit_remaining_support(aggregate: _RemainingTerms) -> None:
    """Cap surviving monomials after signed convolution cancellation."""

    if any(len(terms) > MAX_DIFFERENTIAL_FORM_TERMS for terms in aggregate.values()):
        _output_budget("wedge coefficient support exceeds the bounded output envelope")


def _scalar_unit_sign(form: PolynomialDifferentialForm) -> int | None:
    """Return ``+1`` or ``-1`` when ``form`` is the degree-0 scalar unit."""

    if int(form.degree) != 0 or len(form.components) != 1:
        return None
    component = form.components[0]
    if component.indices:
        return None
    terms = component.coefficient.polynomial.terms
    if len(terms) != 1:
        return None
    term = terms[0]
    coefficient = term.coefficient
    if not _unit_coefficient(coefficient) or any(
        exponent != 0 for exponent in term.exponents
    ):
        return None
    numerator, _denominator = coefficient.as_integer_ratio()
    return numerator


def _negate_rational(value: CanonicalRational) -> CanonicalRational:
    numerator, denominator = value.as_integer_ratio()
    return CanonicalRational.from_integer_ratio(-numerator, denominator)


def _scale_form_by_unit(
    form: PolynomialDifferentialForm, sign: int
) -> PolynomialDifferentialForm:
    """Return ``form`` or ``-form`` according to a scalar unit's sign."""

    if sign == 1:
        return form
    return PolynomialDifferentialForm._from_admitted(
        variables=form.variables,
        degree=form.degree,
        components=tuple(
            FormComponent(
                indices=component.indices,
                coefficient=RationalPolynomial(
                    variables=component.coefficient.variables,
                    polynomial=SparseRationalPolynomial(
                        terms=tuple(
                            RationalPolynomialTerm(
                                coefficient=_negate_rational(term.coefficient),
                                exponents=term.exponents,
                            )
                            for term in component.coefficient.polynomial.terms
                        )
                    ),
                ),
            )
            for component in form.components
        ),
    )


def _admit_form(
    value: object, *, location: tuple[str, ...]
) -> PolynomialDifferentialForm:
    """Admit a native operand without dumping or replaying coefficient budgets."""

    if not isinstance(value, PolynomialDifferentialForm):
        raise OperationDomainValidationError(
            location=location,
            code="differential_form.operand_type",
            message="wedge operands must be polynomial differential forms",
        )
    request_checkpoint("during differential wedge operand admission")
    degree = getattr(value, "degree", None)
    if not isinstance(degree, int) or degree < 0:
        raise OperationDomainValidationError(
            location=(*location, "degree"),
            code="differential_form.degree",
            message="wedge operands must have a nonnegative degree",
        )
    variables = getattr(value, "variables", None)
    if not isinstance(variables, tuple) or len(variables) > MAX_POLYNOMIAL_VARIABLES:
        raise OperationDomainValidationError(
            location=(*location, "variables"),
            code="differential_form.variable_axis",
            message="wedge operands must use a bounded ordered variable axis",
        )
    components = getattr(value, "components", None)
    if not isinstance(components, tuple):
        raise OperationDomainValidationError(
            location=(*location, "components"),
            code="differential_form.component_order",
            message="wedge operands must carry a tuple of form components",
        )
    if len(components) > MAX_DIFFERENTIAL_FORM_COMPONENTS:
        raise OperationDomainValidationError(
            location=(*location, "components"),
            code="differential_form.component_order",
            message="wedge operands exceed the bounded component envelope",
        )
    dimension = len(variables)
    for completed, component in enumerate(components, start=1):
        if completed % _CONVOLUTION_CHECKPOINT_INTERVAL == 0:
            request_checkpoint("during differential wedge operand admission")
        if not isinstance(component, FormComponent):
            raise OperationDomainValidationError(
                location=(*location, "components"),
                code="differential_form.operand_type",
                message="wedge components must be polynomial form components",
            )
        indices = component.indices
        if not isinstance(indices, tuple) or len(indices) > MAX_POLYNOMIAL_VARIABLES:
            raise OperationDomainValidationError(
                location=(*location, "components", "indices"),
                code="differential_form.component_basis",
                message="component indices must be a bounded increasing tuple",
            )
        if degree <= dimension and len(indices) != degree:
            raise OperationDomainValidationError(
                location=(*location, "components", "indices"),
                code="differential_form.component_basis",
                message="component indices must match the form degree on the variable axis",
            )
        coefficient = component.coefficient
        if not isinstance(coefficient, RationalPolynomial):
            raise OperationDomainValidationError(
                location=(*location, "components", "coefficient"),
                code="differential_form.coefficient_axis",
                message="every form coefficient must be a sparse rational polynomial",
            )
        terms = coefficient.polynomial.terms
        if len(terms) > MAX_DIFFERENTIAL_FORM_TERMS:
            raise OperationDomainValidationError(
                location=(*location, "components", "coefficient"),
                code="differential_form.coefficient_budget",
                message="wedge operands exceed the bounded coefficient-term envelope",
            )
        for term_count, _term in enumerate(terms, start=1):
            if term_count % _CONVOLUTION_CHECKPOINT_INTERVAL == 0:
                request_checkpoint("during differential wedge operand admission")
    return value


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


def _admitted_result(
    *,
    variables: tuple[PolynomialVariable, ...],
    degree: int,
    aggregate: _RemainingTerms,
) -> PolynomialDifferentialForm:
    request_checkpoint("during differential wedge result construction")
    components: list[FormComponent] = []
    for indices in sorted(aggregate):
        terms = aggregate[indices]
        if not terms:
            continue
        polynomial_coefficient = RationalPolynomial(
            variables=variables,
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
    return PolynomialDifferentialForm._from_admitted(
        variables=variables, degree=degree, components=tuple(components)
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
        return _zero_form(left.variables, degree)
    if int(left.degree) % 2 == 1 and left == right:
        # Graded commutativity forces an odd form's self-wedge to vanish
        # before exponent or coefficient expansion.
        return _zero_form(left.variables, degree)
    right_unit = _scalar_unit_sign(right)
    if right_unit is not None:
        return _scale_form_by_unit(left, right_unit)
    left_unit = _scalar_unit_sign(left)
    if left_unit is not None:
        return _scale_form_by_unit(right, left_unit)
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
    if term_pair_count > MAX_WEDGE_TERM_PAIRS:
        _term_budget()
    aggregate = _reduce_contributions(_collect_contributions(pairs))
    _admit_remaining_support(aggregate)
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
    return _admitted_result(
        variables=left.variables, degree=degree, aggregate=aggregate
    )


__all__ = ["wedge"]
