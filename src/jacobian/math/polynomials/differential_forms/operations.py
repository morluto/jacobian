"""Exact sparse exterior calculus of polynomial differential forms."""

from __future__ import annotations

from fractions import Fraction

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
    Distinct denominators may share algebraic factors, so the exact reduced
    sum is computed and its actual denominator measured rather than rejecting
    on an unreduced LCM estimate.
    """

    if not current:
        return value
    if not value:
        return current
    total = current + value
    if _integer_decimal_digits(total.denominator) > (
        MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS
    ):
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
            _rational_height_digits(term_coefficient)
            > MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS
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


def _admitted_result(
    *,
    variables: tuple[PolynomialVariable, ...],
    degree: int,
    aggregate: _RemainingTerms,
) -> PolynomialDifferentialForm:
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
        variables=left.variables, degree=degree, aggregate=aggregate
    )


MAX_CALCULUS_TERM_PAIRS = 1_000_000


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
    if value.variables != variables:
        raise OperationDomainValidationError(
            location=location,
            code="differential_form.field_axis",
            message="a vector field must share the form's ordered variable axis",
        )
    return value


def _admit_map(
    value: object, *, location: tuple[str, ...]
) -> PolynomialMap:
    """Admit a native polynomial map without expanding any composition."""

    if not isinstance(value, PolynomialMap):
        raise OperationDomainValidationError(
            location=location,
            code="differential_form.map_type",
            message="a pullback map must be a polynomial map value",
        )
    return value


def _partial_terms(
    terms: tuple[RationalPolynomialTerm, ...], variable: int, dimension: int
) -> dict[tuple[int, ...], Fraction]:
    """Differentiate sparse polynomial terms with respect to one variable."""

    derived: dict[tuple[int, ...], Fraction] = {}
    for term in terms:
        exponent = term.exponents[variable]
        if not exponent:
            continue
        scaled = term.coefficient.as_fraction() * exponent
        if not scaled:
            continue
        exponents = list(term.exponents)
        exponents[variable] -= 1
        key = tuple(exponents)
        derived[key] = derived.get(key, Fraction()) + scaled
    return {key: value for key, value in derived.items() if value}


def _add_monomials(
    aggregate: dict[tuple[int, ...], dict[tuple[int, ...], Fraction]],
    indices: tuple[int, ...],
    monomials: dict[tuple[int, ...], Fraction],
    sign: int,
    *,
    location: tuple[str, ...],
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
    for exponents, value in monomials.items():
        if any(exponent > MAX_DIFFERENTIAL_FORM_EXPONENT for exponent in exponents):
            raise _calculus_budget(
                location,
                "differential_form.calculus.exponent_budget",
                "exterior-calculus exponents exceed the bounded output envelope",
            )
        combined = terms.get(exponents, Fraction()) + sign * value
        if combined:
            terms[exponents] = combined
        elif exponents in terms:
            del terms[exponents]


def _poly_mul(
    left: dict[tuple[int, ...], Fraction],
    right: dict[tuple[int, ...], Fraction],
    *,
    location: tuple[str, ...],
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
    product: dict[tuple[int, ...], Fraction] = {}
    completed = 0
    for left_exponents, left_value in left.items():
        for right_exponents, right_value in right.items():
            completed += 1
            if completed % _CONVOLUTION_CHECKPOINT_INTERVAL == 0:
                request_checkpoint("during differential calculus convolution")
            exponents = tuple(
                a + b
                for a, b in zip(left_exponents, right_exponents, strict=True)
            )
            if any(exponent > MAX_DIFFERENTIAL_FORM_EXPONENT for exponent in exponents):
                raise _calculus_budget(
                    location,
                    "differential_form.calculus.exponent_budget",
                    "exterior-calculus exponents exceed the bounded output envelope",
                )
            product[exponents] = product.get(exponents, Fraction()) + (
                left_value * right_value
            )
    return {key: value for key, value in product.items() if value}


def _poly_pow(
    base: dict[tuple[int, ...], Fraction],
    exponent: int,
    dimension: int,
    *,
    location: tuple[str, ...],
) -> dict[tuple[int, ...], Fraction]:
    """Raise a sparse polynomial to a nonnegative power by squaring."""

    result = {tuple(0 for _ in range(dimension)): Fraction(1)}
    factor = base
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _poly_mul(result, factor, location=location)
        remaining >>= 1
        if remaining:
            factor = _poly_mul(factor, factor, location=location)
    return result


def _substitute_polynomial(
    terms: tuple[RationalPolynomialTerm, ...],
    images: tuple[dict[tuple[int, ...], Fraction], ...],
    dimension: int,
    *,
    location: tuple[str, ...],
) -> dict[tuple[int, ...], Fraction]:
    """Compose a target polynomial with source-axis image polynomials."""

    composed: dict[tuple[int, ...], Fraction] = {}
    for term in terms:
        contribution = _poly_pow(
            {tuple(0 for _ in range(dimension)): term.coefficient.as_fraction()},
            1,
            dimension,
            location=location,
        )
        for variable, exponent in enumerate(term.exponents):
            if exponent:
                contribution = _poly_mul(
                    contribution,
                    _poly_pow(images[variable], exponent, dimension, location=location),
                    location=location,
                )
        for exponents, value in contribution.items():
            composed[exponents] = composed.get(exponents, Fraction()) + value
    return {key: value for key, value in composed.items() if value}


def _partial_dict(
    image: dict[tuple[int, ...], Fraction], variable: int
) -> dict[tuple[int, ...], Fraction]:
    """Differentiate a monomial dictionary with respect to one variable."""

    derived: dict[tuple[int, ...], Fraction] = {}
    for exponents, value in image.items():
        exponent = exponents[variable]
        if not exponent:
            continue
        scaled = value * exponent
        if not scaled:
            continue
        shifted = list(exponents)
        shifted[variable] -= 1
        key = tuple(shifted)
        derived[key] = derived.get(key, Fraction()) + scaled
    return {key: value for key, value in derived.items() if value}


def _differential_of_image(
    image: dict[tuple[int, ...], Fraction],
    dimension: int,
) -> list[dict[tuple[int, ...], Fraction]]:
    """Return the partial derivatives of one image polynomial per source variable."""

    return [_partial_dict(image, variable) for variable in range(dimension)]


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
    aggregate: _RemainingTerms = {}
    location = ("form",)
    for component in form.components:
        for variable in range(dimension):
            if variable in component.indices:
                continue
            derived = _partial_terms(
                component.coefficient.polynomial.terms, variable, dimension
            )
            if not derived:
                continue
            position = sum(1 for index in component.indices if index < variable)
            merged = tuple(sorted((*component.indices, variable)))
            _add_monomials(
                aggregate, merged, derived, -1 if position % 2 else 1,
                location=location,
            )
    for terms in aggregate.values():
        for value in terms.values():
            if (
                _fraction_component_digits(value)
                > MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS
            ):
                raise _calculus_budget(
                    location,
                    "differential_form.calculus.coefficient_budget",
                    "exterior-derivative coefficients exceed the bounded output envelope",
                )
    _admit_remaining_support(aggregate)
    return _admitted_result(
        variables=form.variables, degree=degree, aggregate=aggregate
    )


def interior_product(
    field: PolynomialVectorField,
    form: PolynomialDifferentialForm,
) -> PolynomialDifferentialForm:
    """Contract a polynomial form with a polynomial vector field."""

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
    aggregate: _RemainingTerms = {}
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
            )
            remaining = tuple(
                index for index in component.indices if index != variable
            )
            _add_monomials(
                aggregate, remaining, contracted, -1 if position % 2 else 1,
                location=location,
            )
    _admit_remaining_support(aggregate)
    return _admitted_result(
        variables=form.variables, degree=degree, aggregate=aggregate
    )


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
    images = tuple(
        {
            term.exponents: term.coefficient.as_fraction()
            for term in image.polynomial.terms
        }
        for image in mapping.images
    )
    differentials = [_differential_of_image(image, source_dimension) for image in images]
    aggregate: _RemainingTerms = {}
    location = ("map", "form")
    for component in form.components:
        substituted = _substitute_polynomial(
            component.coefficient.polynomial.terms,
            images,
            source_dimension,
            location=location,
        )
        factors = [differentials[index] for index in component.indices]
        expanded: dict[tuple[int, ...], dict[tuple[int, ...], Fraction]] = {
            (): substituted
        }
        for partials in factors:
            staged: dict[tuple[int, ...], dict[tuple[int, ...], Fraction]] = {}
            for chosen, accumulated in expanded.items():
                for variable in range(source_dimension):
                    contribution = _poly_mul(
                        accumulated, partials[variable], location=location
                    )
                    if contribution:
                        staged[(*chosen, variable)] = contribution
            expanded = staged
            if len(expanded) > MAX_CALCULUS_TERM_PAIRS:
                raise _calculus_budget(
                    location,
                    "differential_form.calculus.term_budget",
                    "pullback differential expansion exceeds the bounded work envelope",
                )
        for chosen, contribution in expanded.items():
            if len(set(chosen)) != len(chosen):
                continue
            inversions = sum(
                1
                for left in range(len(chosen))
                for right in range(left + 1, len(chosen))
                if chosen[left] > chosen[right]
            )
            merged = tuple(sorted(chosen))
            _add_monomials(
                aggregate, merged, contribution, -1 if inversions % 2 else 1,
                location=location,
            )
    _admit_remaining_support(aggregate)
    return _admitted_result(
        variables=mapping.source_variables,
        degree=int(form.degree),
        aggregate=aggregate,
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
    aggregate: _RemainingTerms = {}
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
            )
    _admit_remaining_support(aggregate)
    return _admitted_result(
        variables=left.variables, degree=int(left.degree), aggregate=aggregate
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
    aggregate: _RemainingTerms = {}
    location = ("form",)
    for component in form.components:
        for position, variable in enumerate(component.indices):
            remaining = tuple(index for index in component.indices if index != variable)
            weighted: dict[tuple[int, ...], Fraction] = {}
            for term in component.coefficient.polynomial.terms:
                total = sum(term.exponents) + degree
                value = term.coefficient.as_fraction() / total
                shifted = list(term.exponents)
                shifted[variable] += 1
                key = tuple(shifted)
                weighted[key] = weighted.get(key, Fraction()) + value
            _add_monomials(
                aggregate, remaining, weighted, -1 if position % 2 else 1,
                location=location,
            )
    _admit_remaining_support(aggregate)
    return PrimitiveResult._from_kernel(
        source=form,
        outcome="CONSTRUCTED",
        primitive=_admitted_result(
            variables=form.variables, degree=degree - 1, aggregate=aggregate
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
