"""Exact sparse exterior calculus of polynomial differential forms."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

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
    PolynomialMap,
    PolynomialVectorField,
    PrimitiveResult,
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
_COEFFICIENT_MAGNITUDE_LIMIT = 10**MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS


def _integer_decimal_digit_bound(value: int) -> int:
    """Conservative work proxy, not an exact coefficient-carrier cutoff."""

    magnitude = abs(value)
    if magnitude < 10:
        return 1
    return (magnitude.bit_length() * 30103) // 100000 + 1


def _rational_height_digits(value: CanonicalRational) -> int:
    numerator, denominator = value.as_integer_ratio()
    return max(
        _integer_decimal_digit_bound(abs(numerator)),
        _integer_decimal_digit_bound(denominator),
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
    Distinct denominators may share algebraic factors, so the exact reduced
    sum is computed and its actual denominator measured rather than rejecting
    on an unreduced LCM estimate.
    """

    if not current:
        return value
    if not value:
        return current
    total = current + value
    if total.denominator >= _COEFFICIENT_MAGNITUDE_LIMIT:
        _coefficient_budget()
    return total


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
    for first, second, indices, sign in pairs:
        terms = grouped.setdefault(indices, {})
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


def _rational_scalar_multiple(
    left: PolynomialDifferentialForm, right: PolynomialDifferentialForm
) -> Fraction | None:
    """Return the nonzero rational ``c`` with ``right == c * left``, if any.

    Odd-degree forms satisfy ``alpha wedge (c alpha) == c (alpha wedge alpha)
    == 0``, so a bounded rational-constant proportionality check can cancel a
    zero product before the weighted work preflight charges its products.  The
    check scans the matched component and term support once and never expands a
    product, so it stays inside the admitted operand envelopes.
    """

    if len(left.components) != len(right.components) or not left.components:
        return None
    factor: Fraction | None = None
    for left_component, right_component in zip(
        left.components, right.components, strict=True
    ):
        if left_component.indices != right_component.indices:
            return None
        left_terms = left_component.coefficient.polynomial.terms
        right_terms = right_component.coefficient.polynomial.terms
        if len(left_terms) != len(right_terms):
            return None
        for left_term, right_term in zip(left_terms, right_terms, strict=True):
            if left_term.exponents != right_term.exponents:
                return None
            left_value = left_term.coefficient.as_fraction()
            right_value = right_term.coefficient.as_fraction()
            if not left_value:
                return None
            ratio = right_value / left_value
            if factor is None:
                factor = ratio
            elif ratio != factor:
                return None
    return factor


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
        _admit_component(
            component,
            variables=variables,
            degree=degree,
            dimension=dimension,
            location=location,
        )
    return value


def _admit_component(
    component: object,
    *,
    variables: tuple[str, ...],
    degree: int,
    dimension: int,
    location: tuple[str, ...],
) -> None:
    """Admit one native form component's basis and coefficient carrier."""

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
    if (
        any(type(index) is not int for index in indices)
        or indices != tuple(sorted(set(indices)))
        or any(index < 0 or index >= dimension for index in indices)
    ):
        raise OperationDomainValidationError(
            location=(*location, "components", "indices"),
            code="differential_form.component_basis",
            message=(
                "component indices must be unique, strictly increasing, and lie "
                "on the variable axis"
            ),
        )
    coefficient = component.coefficient
    if not isinstance(coefficient, RationalPolynomial):
        raise OperationDomainValidationError(
            location=(*location, "components", "coefficient"),
            code="differential_form.coefficient_axis",
            message="every form coefficient must be a sparse rational polynomial",
        )
    if coefficient.variables != variables:
        raise OperationDomainValidationError(
            location=(*location, "components", "coefficient"),
            code="differential_form.coefficient_axis",
            message=(
                "every form coefficient must share the form's ordered variable axis"
            ),
        )
    terms = coefficient.polynomial.terms
    if len(terms) > MAX_DIFFERENTIAL_FORM_TERMS:
        raise OperationDomainValidationError(
            location=(*location, "components", "coefficient"),
            code="differential_form.coefficient_budget",
            message="wedge operands exceed the bounded coefficient-term envelope",
        )
    for term_count, term in enumerate(terms, start=1):
        if term_count % _CONVOLUTION_CHECKPOINT_INTERVAL == 0:
            request_checkpoint("during differential wedge operand admission")
        term_coefficient = getattr(term, "coefficient", None)
        term_exponents = getattr(term, "exponents", None)
        if (
            not isinstance(term_coefficient, CanonicalRational)
            or not isinstance(term_exponents, tuple)
            or any(type(exponent) is not int for exponent in term_exponents)
            or len(term_exponents) != dimension
        ):
            raise OperationDomainValidationError(
                location=(*location, "components", "coefficient"),
                code="differential_form.coefficient_shape",
                message="wedge coefficient terms must use the bounded variable axis",
            )
        if any(
            exponent < 0 or exponent > MAX_DIFFERENTIAL_FORM_EXPONENT
            for exponent in term_exponents
        ):
            raise OperationDomainValidationError(
                location=(*location, "components", "coefficient"),
                code="differential_form.coefficient_exponent",
                message="wedge coefficient exponents exceed the bounded output envelope",
            )
        numerator, denominator = term_coefficient.as_integer_ratio()
        if (
            abs(numerator) >= _COEFFICIENT_MAGNITUDE_LIMIT
            or denominator >= _COEFFICIENT_MAGNITUDE_LIMIT
        ):
            raise OperationDomainValidationError(
                location=(*location, "components", "coefficient"),
                code="differential_form.coefficient_height",
                message="wedge coefficient height exceeds the bounded output envelope",
            )
        reduced = (
            Fraction(numerator, denominator)
            if type(numerator) is int and type(denominator) is int and denominator > 0
            else None
        )
        if reduced is None or (
            reduced.numerator,
            reduced.denominator,
        ) != (numerator, denominator):
            raise OperationDomainValidationError(
                location=(*location, "components", "coefficient"),
                code="differential_form.coefficient_canonical",
                message=(
                    "wedge coefficient terms must carry reduced rationals with a "
                    "positive denominator"
                ),
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


def _admit_final_coefficients(
    aggregate: _RemainingTerms, *, location: tuple[str, ...]
) -> None:
    """Prove the carrier and bounded construction cost after exact cancellation."""

    if len(aggregate) > MAX_DIFFERENTIAL_FORM_COMPONENTS or any(
        len(terms) > MAX_DIFFERENTIAL_FORM_TERMS for terms in aggregate.values()
    ):
        raise _calculus_budget(
            location,
            "differential_form.calculus.output_budget",
            "form result exceeds its component or term envelope",
        )
    digits = 0
    for terms in aggregate.values():
        request_checkpoint("during differential-form result admission")
        for value in terms.values():
            if (
                abs(value.numerator) >= _COEFFICIENT_MAGNITUDE_LIMIT
                or value.denominator >= _COEFFICIENT_MAGNITUDE_LIMIT
            ):
                raise _calculus_budget(
                    location,
                    "differential_form.calculus.coefficient_budget",
                    "form result exceeds its exact coefficient-height carrier",
                )
            digits += _fraction_component_digits(value)
            if digits > MAX_CALCULUS_STORAGE_DIGITS:
                raise _calculus_budget(
                    location,
                    "differential_form.calculus.output_budget",
                    "form result exceeds its scalar-construction budget",
                )


def _admitted_result(
    *,
    variables: tuple[PolynomialVariable, ...],
    degree: int,
    aggregate: _RemainingTerms,
    location: tuple[str, ...] = (),
) -> PolynomialDifferentialForm:
    _admit_final_coefficients(aggregate, location=location)
    request_checkpoint("during differential wedge result construction")
    components: list[FormComponent] = []
    completed = 0
    for indices in sorted(aggregate):
        terms = aggregate[indices]
        if not terms:
            continue
        materialized: list[RationalPolynomialTerm] = []
        for exponents, value in sorted(terms.items(), reverse=True):
            completed += 1
            if completed % _CONVOLUTION_CHECKPOINT_INTERVAL == 0:
                request_checkpoint("during differential wedge result construction")
            materialized.append(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(value),
                    exponents=exponents,
                )
            )
        polynomial_coefficient = RationalPolynomial(
            variables=variables,
            polynomial=SparseRationalPolynomial(terms=tuple(materialized)),
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
    # Graded commutativity forces an odd form's wedge with any rational
    # multiple of itself (including its own self-wedge) to vanish before
    # exponent or coefficient expansion.
    if int(left.degree) % 2 == 1 and _rational_scalar_multiple(left, right) is not None:
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
    # Preflight the weighted digit work from operand term heights before
    # multiplying or retaining any product.
    digit_work = 0
    completed = 0
    for first, second, _, _ in pairs:
        left_heights = tuple(
            _rational_height_digits(term.coefficient)
            for term in first.coefficient.polynomial.terms
        )
        right_heights = tuple(
            _rational_height_digits(term.coefficient)
            for term in second.coefficient.polynomial.terms
        )
        for left_digits in left_heights:
            for right_digits in right_heights:
                completed += 1
                if completed % _CONVOLUTION_CHECKPOINT_INTERVAL == 0:
                    request_checkpoint("during differential wedge work preflight")
                digit_work += left_digits * right_digits
                if digit_work > MAX_WEDGE_DIGIT_WORK:
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
        variables=left.variables,
        degree=degree,
        aggregate=aggregate,
        location=("left", "right"),
    )


MAX_CALCULUS_TERM_PAIRS = 1_000_000
# Up to eight alternating factors need a larger coupled work envelope than
# one binary wedge; support and retained scalar height remain independently bounded.
MAX_CALCULUS_DIGIT_WORK = 100_000_000_000
MAX_CALCULUS_STORAGE_DIGITS = 16_000_000
MAX_CALCULUS_INTERMEDIATE_DIGITS = 131_072


def _calculus_budget(
    location: tuple[str, ...], code: str, message: str
) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=location, code=code, message=message
    )


def _admit_vector_field(
    value: object, variables: tuple[str, ...], *, location: tuple[str, ...]
) -> PolynomialVectorField:
    """Admit a native polynomial vector field on the caller's variable axis."""

    if not isinstance(value, PolynomialVectorField):
        raise OperationDomainValidationError(
            location=location,
            code="differential_form.field_type",
            message="a vector field must be a polynomial vector field value",
        )
    if getattr(value, "variables", None) != variables:
        raise OperationDomainValidationError(
            location=location,
            code="differential_form.field_axis",
            message="a vector field must share the form's ordered variable axis",
        )
    # The field has the same bounded polynomial family as an endomorphism
    # of its axis. Reuse native shape/scalar admission before Fraction work
    # or the scalar-contraction zero shortcut.
    try:
        admitted = _admit_map(
            PolynomialMap.model_construct(
                source_variables=variables,
                target_variables=variables,
                images=getattr(value, "components", None),
            ),
            location=location,
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError as exc:
        raise OperationDomainValidationError(
            location=location,
            code="differential_form.field_shape",
            message="contraction requires a canonical bounded polynomial vector field",
        ) from exc
    return PolynomialVectorField.model_construct(
        variables=variables, components=admitted.images
    )


def _admit_map(value: object, *, location: tuple[str, ...]) -> PolynomialMap:
    """Admit a native polynomial map without expanding any composition."""

    if not isinstance(value, PolynomialMap):
        raise OperationDomainValidationError(
            location=location,
            code="differential_form.map_type",
            message="a pullback map must be a polynomial map value",
        )
    try:
        # Bound native containers before materialization. In particular, the
        # dimensional-zero shortcut must not certify malformed map objects.
        axes = (value.source_variables, value.target_variables)
        if any(
            not isinstance(axis, tuple)
            or len(axis) > MAX_POLYNOMIAL_VARIABLES
            or any(type(name) is not str or len(name) > 32 for name in axis)
            for axis in axes
        ):
            raise ValueError("invalid map axes")
        if not isinstance(value.images, tuple) or len(value.images) != len(
            value.target_variables
        ):
            raise ValueError("invalid image count")
        for image in value.images:
            request_checkpoint("during pullback map admission")
            if (
                not isinstance(image, RationalPolynomial)
                or type(image.domain) is not str
                or image.domain != "QQ"
                or image.variables != value.source_variables
                or not isinstance(image.polynomial, SparseRationalPolynomial)
                or not isinstance(image.polynomial.terms, tuple)
                or len(image.polynomial.terms) > MAX_DIFFERENTIAL_FORM_TERMS
            ):
                raise ValueError("invalid map image")
            for term in image.polynomial.terms:
                if (
                    not isinstance(term, RationalPolynomialTerm)
                    or not isinstance(term.exponents, tuple)
                    or len(term.exponents) != len(value.source_variables)
                    or any(
                        type(e) is not int
                        or not 0 <= e <= MAX_DIFFERENTIAL_FORM_EXPONENT
                        for e in term.exponents
                    )
                    or not isinstance(term.coefficient, CanonicalRational)
                ):
                    raise ValueError("invalid image term")
                if any(
                    type(scalar) is not int
                    or scalar.bit_length()
                    > 4 * MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS
                    for scalar in (term.coefficient.num, term.coefficient.den)
                ):
                    raise ValueError("invalid image coefficient")
        # Structural revalidation enforces exact decimal heights, reduced
        # coefficients, term ordering and axes; it performs no substitution.
        return PolynomialMap.model_validate(value.model_dump())
    except (ValidationError, ValueError, TypeError, AttributeError) as exc:
        raise OperationDomainValidationError(
            location=location,
            code="differential_form.map_shape",
            message="pullback requires a canonical bounded polynomial map",
        ) from exc


def _partial_terms(
    terms: tuple[RationalPolynomialTerm, ...],
    variable: int,
    dimension: int,
    *,
    arithmetic: _CalculusArithmeticBudget,
    location: tuple[str, ...],
) -> dict[tuple[int, ...], Fraction]:
    """Differentiate sparse polynomial terms with respect to one variable."""

    arithmetic.charge_pairs(len(terms), location=location)
    derived: dict[tuple[int, ...], Fraction] = {}
    for term in terms:
        exponent = term.exponents[variable]
        if not exponent:
            continue
        scaled = arithmetic.multiply(
            term.coefficient.as_fraction(), Fraction(exponent), location=location
        )
        if not scaled:
            continue
        exponents = list(term.exponents)
        exponents[variable] -= 1
        key = tuple(exponents)
        derived[key] = arithmetic.add(
            derived.get(key, Fraction()), scaled, location=location
        )
    return {key: value for key, value in derived.items() if value}


def _add_monomials(
    aggregate: dict[tuple[int, ...], dict[tuple[int, ...], Fraction]],
    indices: tuple[int, ...],
    monomials: dict[tuple[int, ...], Fraction],
    sign: int,
    *,
    location: tuple[str, ...],
    budget: _CalculusArithmeticBudget,
    storage: _AggregateStorage,
) -> None:
    """Fold signed monomials into a form aggregate with a term-count preflight."""

    if not monomials:
        return
    terms = aggregate.setdefault(indices, {})
    if len(terms) + len(monomials) > MAX_CALCULUS_TERM_PAIRS:
        raise _calculus_budget(
            location,
            "differential_form.calculus.term_budget",
            "exterior-calculus monomial support exceeds the bounded work envelope",
        )
    budget.charge_pairs(len(monomials), location=location)
    for exponents, value in monomials.items():
        if any(exponent > MAX_DIFFERENTIAL_FORM_EXPONENT for exponent in exponents):
            raise _calculus_budget(
                location,
                "differential_form.calculus.exponent_budget",
                "exterior-calculus exponents exceed the bounded output envelope",
            )
        current = terms.get(exponents, Fraction())
        combined = budget.add(current, sign * value, location=location)
        storage.digits += (budget.height(combined) if combined else 0) - (
            budget.height(current) if current else 0
        )
        budget.check_storage(storage.digits, location=location)
        if combined:
            terms[exponents] = combined
        elif exponents in terms:
            del terms[exponents]


def _poly_mul(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
    *,
    location: tuple[str, ...],
    budget: _CalculusArithmeticBudget,
) -> dict[tuple[int, ...], Fraction]:
    """Multiply sparse monomial dictionaries with a pair-count preflight."""

    if not left or not right:
        return {}
    if len(left) * len(right) > MAX_CALCULUS_TERM_PAIRS:
        raise _calculus_budget(
            location,
            "differential_form.calculus.term_budget",
            "exterior-calculus polynomial convolution exceeds the bounded work envelope",
        )
    budget.charge_pairs(len(left) * len(right), location=location)
    product: dict[tuple[int, ...], Fraction] = {}
    completed = 0
    retained_digits = 0
    for left_exponents, left_value in left.items():
        for right_exponents, right_value in right.items():
            completed += 1
            if completed % _CONVOLUTION_CHECKPOINT_INTERVAL == 0:
                request_checkpoint("during differential calculus convolution")
            exponents = tuple(
                a + b for a, b in zip(left_exponents, right_exponents, strict=True)
            )
            if any(exponent > MAX_DIFFERENTIAL_FORM_EXPONENT for exponent in exponents):
                raise _calculus_budget(
                    location,
                    "differential_form.calculus.exponent_budget",
                    "exterior-calculus exponents exceed the bounded output envelope",
                )
            current = product.get(exponents, Fraction())
            value = budget.multiply(left_value, right_value, location=location)
            combined = budget.add(current, value, location=location)
            retained_digits += (budget.height(combined) if combined else 0) - (
                budget.height(current) if current else 0
            )
            budget.check_storage(retained_digits, location=location)
            product[exponents] = combined
    return {key: value for key, value in product.items() if value}


def _poly_pow(
    base: dict[tuple[int, ...], Fraction],
    exponent: int,
    dimension: int,
    *,
    location: tuple[str, ...],
    budget: _CalculusArithmeticBudget,
) -> dict[tuple[int, ...], Fraction]:
    """Raise a sparse polynomial to a nonnegative power by squaring."""

    result = {tuple(0 for _ in range(dimension)): Fraction(1)}
    factor = base
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _poly_mul(result, factor, location=location, budget=budget)
        remaining >>= 1
        if remaining:
            factor = _poly_mul(factor, factor, location=location, budget=budget)
    return result


def _substitute_polynomial(
    terms: tuple[RationalPolynomialTerm, ...],
    images: tuple[dict[tuple[int, ...], Fraction], ...],
    dimension: int,
    *,
    location: tuple[str, ...],
    budget: _CalculusArithmeticBudget,
) -> dict[tuple[int, ...], Fraction]:
    """Compose a target polynomial with source-axis image polynomials."""

    composed: dict[tuple[int, ...], Fraction] = {}
    composed_storage = _AggregateStorage()
    for term in terms:
        contribution = _poly_pow(
            {tuple(0 for _ in range(dimension)): term.coefficient.as_fraction()},
            1,
            dimension,
            location=location,
            budget=budget,
        )
        for variable, exponent in enumerate(term.exponents):
            if exponent:
                contribution = _poly_mul(
                    contribution,
                    _poly_pow(
                        images[variable],
                        exponent,
                        dimension,
                        location=location,
                        budget=budget,
                    ),
                    location=location,
                    budget=budget,
                )
        _add_monomials(
            {(): composed},
            (),
            contribution,
            1,
            location=location,
            budget=budget,
            storage=composed_storage,
        )
    return {key: value for key, value in composed.items() if value}


def _partial_dict(
    image: dict[tuple[int, ...], Fraction],
    variable: int,
    *,
    arithmetic: _CalculusArithmeticBudget,
    location: tuple[str, ...],
) -> dict[tuple[int, ...], Fraction]:
    """Differentiate a monomial dictionary with respect to one variable."""

    arithmetic.charge_pairs(len(image), location=location)
    derived: dict[tuple[int, ...], Fraction] = {}
    for exponents, value in image.items():
        exponent = exponents[variable]
        if not exponent:
            continue
        scaled = arithmetic.multiply(value, Fraction(exponent), location=location)
        if not scaled:
            continue
        shifted = list(exponents)
        shifted[variable] -= 1
        key = tuple(shifted)
        derived[key] = arithmetic.add(
            derived.get(key, Fraction()), scaled, location=location
        )
    return {key: value for key, value in derived.items() if value}


def _differential_of_image(
    image: dict[tuple[int, ...], Fraction],
    dimension: int,
    *,
    arithmetic: _CalculusArithmeticBudget,
    location: tuple[str, ...],
) -> list[dict[tuple[int, ...], Fraction]]:
    """Return the partial derivatives of one image polynomial per source variable."""

    return [
        _partial_dict(image, variable, arithmetic=arithmetic, location=location)
        for variable in range(dimension)
    ]


def exterior_derivative(
    form: PolynomialDifferentialForm,
) -> PolynomialDifferentialForm:
    """Return the exact exterior derivative ``d`` of a polynomial form."""

    execution_deadline(WEDGE_WALL_SECONDS)
    form = _admit_form(form, location=("form",))
    dimension = len(form.variables)
    degree = int(form.degree) + 1
    _admit_degree(degree)
    if degree > dimension:
        return _zero_form(form.variables, degree)
    arithmetic = _CalculusArithmeticBudget()
    aggregate: _RemainingTerms = {}
    aggregate_storage = _AggregateStorage()
    location = ("form",)
    for component in form.components:
        for variable in range(dimension):
            if variable in component.indices:
                continue
            derived = _partial_terms(
                component.coefficient.polynomial.terms,
                variable,
                dimension,
                arithmetic=arithmetic,
                location=location,
            )
            if not derived:
                continue
            position = sum(1 for index in component.indices if index < variable)
            merged = tuple(sorted((*component.indices, variable)))
            _add_monomials(
                aggregate,
                merged,
                derived,
                -1 if position % 2 else 1,
                location=location,
                budget=arithmetic,
                storage=aggregate_storage,
            )
    _admit_remaining_support(aggregate)
    return _admitted_result(
        variables=form.variables, degree=degree, aggregate=aggregate, location=location
    )


def interior_product(
    field: PolynomialVectorField,
    form: PolynomialDifferentialForm,
) -> PolynomialDifferentialForm:
    """Contract a form; scalar contraction is represented by degree-zero zero."""

    execution_deadline(WEDGE_WALL_SECONDS)
    form = _admit_form(form, location=("form",))
    field = _admit_vector_field(field, form.variables, location=("field",))
    if not int(form.degree):
        return _zero_form(form.variables, 0)
    degree = int(form.degree) - 1
    field_terms = tuple(
        {
            term.exponents: term.coefficient.as_fraction()
            for term in component.polynomial.terms
        }
        for component in field.components
    )
    arithmetic = _CalculusArithmeticBudget()
    aggregate: _RemainingTerms = {}
    aggregate_storage = _AggregateStorage()
    location = ("field", "form")
    for component in form.components:
        for position, variable in enumerate(component.indices):
            contracted = _poly_mul(
                {
                    term.exponents: term.coefficient.as_fraction()
                    for term in component.coefficient.polynomial.terms
                },
                field_terms[variable],
                location=location,
                budget=arithmetic,
            )
            remaining = tuple(index for index in component.indices if index != variable)
            _add_monomials(
                aggregate,
                remaining,
                contracted,
                -1 if position % 2 else 1,
                location=location,
                budget=arithmetic,
                storage=aggregate_storage,
            )
    _admit_remaining_support(aggregate)
    return _admitted_result(
        variables=form.variables, degree=degree, aggregate=aggregate, location=location
    )


@dataclass
class _AggregateStorage:
    """Incremental scalar storage for one live aggregate, without retaining it."""

    digits: int = 0


@dataclass
class _CalculusArithmeticBudget:
    """Request-local rational work and workspace, distinct from final height."""

    pairs: int = 0
    digit_work: int = 0
    code_prefix: str = "differential_form.calculus."
    label: str = "exterior-calculus arithmetic"
    pair_limit: int = MAX_CALCULUS_TERM_PAIRS
    work_limit: int = MAX_CALCULUS_DIGIT_WORK
    storage_limit: int = MAX_CALCULUS_STORAGE_DIGITS

    @staticmethod
    def height(value: Fraction) -> int:
        # A private workspace, larger than the final form carrier. An admitted
        # image derivative has at most 8195 digits. Reduced intermediates
        # are independently capped; the workspace need not admit every
        # formal product. Transient Fraction operands fit 2H+1.
        bits = max(value.numerator.bit_length(), value.denominator.bit_length())
        return (bits * 30103) // 100000 + 1

    def retain(self, value: Fraction, *, location: tuple[str, ...]) -> Fraction:
        if self.height(value) > MAX_CALCULUS_INTERMEDIATE_DIGITS:
            raise _calculus_budget(
                location,
                f"{self.code_prefix}height",
                f"{self.label} exceeds its private scalar-height budget",
            )
        return value

    def charge_scalars(
        self, left: Fraction, right: Fraction, *, location: tuple[str, ...]
    ) -> None:
        self.retain(left, location=location)
        self.retain(right, location=location)
        self.charge_work(self.height(left) * self.height(right), location=location)

    def charge_work(self, amount: int, *, location: tuple[str, ...]) -> None:
        self.digit_work += amount
        if self.digit_work > self.work_limit:
            raise _calculus_budget(
                location,
                f"{self.code_prefix}work",
                f"{self.label} exceeds its cumulative digit-work budget",
            )

    def multiply(
        self, left: Fraction, right: Fraction, *, location: tuple[str, ...]
    ) -> Fraction:
        if left in (-1, 1):
            return self.retain(right if left == 1 else -right, location=location)
        if right in (-1, 1):
            return self.retain(left if right == 1 else -left, location=location)
        self.charge_scalars(left, right, location=location)
        return self.retain(left * right, location=location)

    def add(
        self, left: Fraction, right: Fraction, *, location: tuple[str, ...]
    ) -> Fraction:
        if not left:
            return self.retain(right, location=location)
        if not right:
            return self.retain(left, location=location)
        if left.denominator == right.denominator:
            self.retain(left, location=location)
            self.retain(right, location=location)
            # No denominator cross-products are needed. Price the numerator
            # sum and its reduction against the common denominator; for
            # integers this is linear rather than quadratic in their height.
            numerator_digits = (
                sum(
                    (value.numerator.bit_length() * 30103) // 100000 + 1
                    for value in (left, right)
                )
                + 1
            )
            denominator_digits = (left.denominator.bit_length() * 30103) // 100000 + 1
            self.charge_work(numerator_digits * denominator_digits, location=location)
        else:
            self.charge_scalars(left, right, location=location)
        return self.retain(left + right, location=location)

    def check_storage(self, digits: int, *, location: tuple[str, ...]) -> None:
        # Bound each complete exterior stage and its temporary polynomial.
        # At most the old stage, new stage and one product coexist. Each
        # tracked height bounds max(numerator, denominator) digits, so their
        # combined scalar payload is at most twice this count.
        if digits > self.storage_limit:
            raise _calculus_budget(
                location,
                f"{self.code_prefix}storage",
                f"{self.label} exceeds its scalar-storage budget",
            )

    def charge_pairs(self, amount: int, *, location: tuple[str, ...]) -> None:
        self.pairs += amount
        if self.pairs > self.pair_limit:
            raise _calculus_budget(
                location,
                f"{self.code_prefix}work",
                f"{self.label} exceeds its cumulative pair budget",
            )


def _pullback_differential_factor(
    indices: tuple[int, ...],
    differentials: list[list[dict[tuple[int, ...], Fraction]]],
    dimension: int,
    *,
    location: tuple[str, ...],
    budget: _CalculusArithmeticBudget,
    initial: dict[tuple[int, ...], Fraction] | None = None,
) -> _RemainingTerms:
    """Expand dF_i1 wedge ... wedge dF_ik in the alternating basis first."""

    if initial is not None:
        budget.charge_pairs(len(initial), location=location)
        initial_digits = 0
        for value in initial.values():
            budget.retain(value, location=location)
            initial_digits += budget.height(value)
        budget.check_storage(initial_digits, location=location)
    expanded: _RemainingTerms = {
        (): {(0,) * dimension: Fraction(1)} if initial is None else initial
    }
    for index in indices:
        request_checkpoint("during pullback differential presolve")
        staged: _RemainingTerms = {}
        staged_storage = _AggregateStorage()
        for chosen, accumulated in expanded.items():
            for variable, partial in enumerate(differentials[index]):
                # An alternating factor with a repeated basis index is zero;
                # do not expand polynomial products destined to disappear.
                if variable in chosen or not partial:
                    continue
                contribution = _poly_mul(
                    accumulated, partial, location=location, budget=budget
                )
                inversions = sum(previous > variable for previous in chosen)
                _add_monomials(
                    staged,
                    tuple(sorted((*chosen, variable))),
                    contribution,
                    -1 if inversions % 2 else 1,
                    location=location,
                    budget=budget,
                    storage=staged_storage,
                )
        expanded = {basis: terms for basis, terms in staged.items() if terms}
        if not expanded:
            break
    return expanded


def pullback(
    mapping: PolynomialMap,
    form: PolynomialDifferentialForm,
) -> PolynomialDifferentialForm:
    """Pull a polynomial form back along a polynomial map."""

    execution_deadline(WEDGE_WALL_SECONDS)
    mapping = _admit_map(mapping, location=("map",))
    form = _admit_form(form, location=("form",))
    if form.variables != mapping.target_variables:
        raise OperationDomainValidationError(
            location=("form", "variables"),
            code="differential_form.pullback_axis",
            message="a pulled-back form must use the map target axis",
        )
    source_dimension = len(mapping.source_variables)
    if int(form.degree) > source_dimension or not form.components:
        return _zero_form(mapping.source_variables, int(form.degree))
    arithmetic = _CalculusArithmeticBudget()
    images = tuple(
        {
            term.exponents: term.coefficient.as_fraction()
            for term in image.polynomial.terms
        }
        for image in mapping.images
    )
    differentials = [
        _differential_of_image(
            image, source_dimension, arithmetic=arithmetic, location=("map", "form")
        )
        for image in images
    ]
    # At most 28 pairs of admitted rows (each at most 8*256 terms).
    # Exact repeated differentials have zero exterior product regardless of
    # their coefficient height; recognize this before arithmetic pricing.
    repeated_differentials = {
        (left, right)
        for left, right in combinations(range(len(differentials)), 2)
        if differentials[left] == differentials[right]
    }
    aggregate: _RemainingTerms = {}
    aggregate_storage = _AggregateStorage()
    location = ("map", "form")
    # Reserve two independently bounded orderings for the complete request.
    # Trying coefficient-first after a factor refusal does not reset the
    # deadline or either ledger; total extra work is at most their sum.
    presolve_budget = _CalculusArithmeticBudget(
        code_prefix="differential_form.pullback.presolve_",
        label="pullback differential presolve",
        pair_limit=MAX_CALCULUS_TERM_PAIRS,
        work_limit=MAX_CALCULUS_DIGIT_WORK,
        storage_limit=MAX_CALCULUS_STORAGE_DIGITS,
    )
    coefficient_first_budget = _CalculusArithmeticBudget(
        code_prefix="differential_form.pullback.presolve_",
        label="pullback coefficient-first recovery",
        pair_limit=MAX_CALCULUS_TERM_PAIRS,
        work_limit=MAX_CALCULUS_DIGIT_WORK,
        storage_limit=MAX_CALCULUS_STORAGE_DIGITS,
    )
    for component in form.components:
        if any(
            pair in repeated_differentials
            for pair in combinations(component.indices, 2)
        ):
            continue
        try:
            differential_factor = _pullback_differential_factor(
                component.indices,
                differentials,
                source_dimension,
                location=location,
                budget=presolve_budget,
            )
        except OperationResourceAdmissionError:
            # Release the failed phase's traceback/workspace before recovery.
            differential_factor = None
        if differential_factor is None:
            # A coefficient can cancel scalar growth or vanish identically.
            # Preserve the useful coefficient-first path, but price that
            # alternative separately and attempt it at most once per component.
            substituted = _substitute_polynomial(
                component.coefficient.polynomial.terms,
                images,
                source_dimension,
                location=location,
                budget=arithmetic,
            )
            if not substituted:
                continue
            weighted_factor = _pullback_differential_factor(
                component.indices,
                differentials,
                source_dimension,
                location=location,
                budget=coefficient_first_budget,
                initial=substituted,
            )
            for indices, contribution in weighted_factor.items():
                _add_monomials(
                    aggregate,
                    indices,
                    contribution,
                    1,
                    location=location,
                    budget=arithmetic,
                    storage=aggregate_storage,
                )
            continue
        if not differential_factor:
            continue
        substituted = _substitute_polynomial(
            component.coefficient.polynomial.terms,
            images,
            source_dimension,
            location=location,
            budget=arithmetic,
        )
        for indices, factor in differential_factor.items():
            contribution = _poly_mul(
                substituted, factor, location=location, budget=arithmetic
            )
            _add_monomials(
                aggregate,
                indices,
                contribution,
                1,
                location=location,
                budget=arithmetic,
                storage=aggregate_storage,
            )
    _admit_remaining_support(aggregate)
    return _admitted_result(
        variables=mapping.source_variables,
        degree=int(form.degree),
        aggregate=aggregate,
        location=location,
    )


def lie_derivative(
    field: PolynomialVectorField,
    form: PolynomialDifferentialForm,
) -> PolynomialDifferentialForm:
    """Return the Lie derivative via Cartan's formula ``L = i d + d i``."""

    execution_deadline(WEDGE_WALL_SECONDS)
    form = _admit_form(form, location=("form",))
    field = _admit_vector_field(field, form.variables, location=("field",))
    first = interior_product(field, exterior_derivative(form))
    if not int(form.degree):
        # i_X(f) vanishes in degree -1. Standalone contraction represents it
        # as degree-zero zero, whose derivative has the wrong grade here.
        return first
    second = exterior_derivative(interior_product(field, form))
    return _add_forms(first, second)


def _add_forms(
    left: PolynomialDifferentialForm, right: PolynomialDifferentialForm
) -> PolynomialDifferentialForm:
    """Add two forms on one axis with exact rational combination."""

    if left.variables != right.variables or left.degree != right.degree:
        raise OperationDomainValidationError(
            location=("left", "right"),
            code="differential_form.calculus_axis",
            message="combined forms must share one variable axis and degree",
        )
    arithmetic = _CalculusArithmeticBudget()
    aggregate: _RemainingTerms = {}
    aggregate_storage = _AggregateStorage()
    location = ("left", "right")
    for form, sign in ((left, 1), (right, 1)):
        for component in form.components:
            _add_monomials(
                aggregate,
                component.indices,
                {
                    term.exponents: term.coefficient.as_fraction()
                    for term in component.coefficient.polynomial.terms
                },
                sign,
                location=location,
                budget=arithmetic,
                storage=aggregate_storage,
            )
    _admit_remaining_support(aggregate)
    return _admitted_result(
        variables=left.variables,
        degree=int(left.degree),
        aggregate=aggregate,
        location=location,
    )


def affine_homotopy_primitive(
    form: PolynomialDifferentialForm,
) -> PrimitiveResult:
    """Return the cone-homotopy primitive of a closed positive-degree form."""

    execution_deadline(WEDGE_WALL_SECONDS)
    form = _admit_form(form, location=("form",))
    degree = int(form.degree)
    if not degree:
        return PrimitiveResult._from_kernel(source=form, outcome="NOT_APPLICABLE")
    if exterior_derivative(form).components:
        return PrimitiveResult._from_kernel(source=form, outcome="NOT_APPLICABLE")
    arithmetic = _CalculusArithmeticBudget()
    aggregate: _RemainingTerms = {}
    aggregate_storage = _AggregateStorage()
    location = ("form",)
    for component in form.components:
        for position, variable in enumerate(component.indices):
            remaining = tuple(index for index in component.indices if index != variable)
            weighted: dict[tuple[int, ...], Fraction] = {}
            for term in component.coefficient.polynomial.terms:
                total = sum(term.exponents) + degree
                arithmetic.charge_pairs(1, location=location)
                value = arithmetic.multiply(
                    term.coefficient.as_fraction(),
                    Fraction(1, total),
                    location=location,
                )
                shifted = list(term.exponents)
                shifted[variable] += 1
                key = tuple(shifted)
                weighted[key] = arithmetic.add(
                    weighted.get(key, Fraction()), value, location=location
                )
            _add_monomials(
                aggregate,
                remaining,
                weighted,
                -1 if position % 2 else 1,
                location=location,
                budget=arithmetic,
                storage=aggregate_storage,
            )
    _admit_remaining_support(aggregate)
    return PrimitiveResult._from_kernel(
        source=form,
        outcome="CONSTRUCTED",
        primitive=_admitted_result(
            variables=form.variables,
            degree=degree - 1,
            aggregate=aggregate,
            location=location,
        ),
    )


__all__ = [
    "affine_homotopy_primitive",
    "exterior_derivative",
    "interior_product",
    "lie_derivative",
    "pullback",
    "wedge",
]
