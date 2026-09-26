"""Exact bounded arithmetic on rational Puiseux windows."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from math import ceil, floor, gcd, lcm, log2
from typing import NoReturn

from pydantic import TypeAdapter, ValidationError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.local_series.puiseux_values import (
    MAX_PUISEUX_RAMIFICATION,
    PuiseuxTerm,
    TruncatedPuiseuxWindow,
)
from jacobian.math.polynomials.local_series.values import (
    MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
    MAX_LOCAL_SERIES_EXPONENT,
    MAX_LOCAL_SERIES_TERMS,
)
from jacobian.math.polynomials.values import PolynomialVariable

MAX_PUISEUX_ARITHMETIC_WORK = 1_000_000
# Aggregate serialized-digit envelope for one canonical Puiseux result,
# matching the canonical 10 MiB encoded-output limit. One canonical decimal
# digit prices one ASCII output character, so admission rejects a result
# that could not be transported instead of only the worst-case product of
# the retained-term and coefficient-digit representation bounds.
MAX_PUISEUX_RESULT_DIGITS = 10 * 1024 * 1024
_MAX_SCALAR_BITS = floor(MAX_LOCAL_SERIES_COEFFICIENT_DIGITS * log2(10)) + 1
_VARIABLE_ADAPTER = TypeAdapter(PolynomialVariable)


def _domain(code: str, message: str, location: tuple[str, ...]) -> NoReturn:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _resource(code: str, message: str, location: tuple[str, ...]) -> NoReturn:
    raise OperationResourceAdmissionError(location=location, code=code, message=message)


def _canonical(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(value)


def _check_rational(value: object, *, label: str, exponent: bool = False) -> Fraction:
    if not isinstance(value, CanonicalRational):
        _domain("puiseux_value_type", f"{label} must be a canonical rational", (label,))
    num_raw = getattr(value, "num", None)
    den_raw = getattr(value, "den", None)
    if type(num_raw) is not int or type(den_raw) is not int or den_raw <= 0:
        _domain("puiseux_value", f"{label} must be a valid rational", (label,))
    num: int = num_raw
    den: int = den_raw
    if max(abs(num), den).bit_length() > _MAX_SCALAR_BITS:
        _resource(
            "puiseux_scalar_bound", f"{label} exceeds the scalar digit bound", (label,)
        )
    result = Fraction(num, den)
    if (num, den) != (result.numerator, result.denominator):
        _domain("puiseux_value", f"{label} must be reduced", (label,))
    try:
        require_bounded_rational(
            value,
            max_digits=MAX_LOCAL_SERIES_COEFFICIENT_DIGITS,
            label=label,
        )
    except ValueError as error:
        _resource("puiseux_scalar_bound", str(error), (label,))
    if exponent and abs(result) > MAX_LOCAL_SERIES_EXPONENT:
        _resource(
            "puiseux_exponent_bound", f"{label} exceeds the exponent bound", (label,)
        )
    return result


def _check(
    window: TruncatedPuiseuxWindow,
) -> tuple[Fraction, Fraction, tuple[tuple[Fraction, Fraction], ...], Fraction]:
    if not isinstance(window, TruncatedPuiseuxWindow):
        _domain(
            "puiseux_series_type",
            "series must be a truncated Puiseux window",
            ("series",),
        )
    if type(window.variable) is not str:
        _domain(
            "puiseux_variable",
            "variable must be a strict identifier",
            ("series", "variable"),
        )
    try:
        _VARIABLE_ADAPTER.validate_python(window.variable, strict=True)
    except ValidationError as error:
        raise OperationDomainValidationError(
            location=("series", "variable"),
            code="local_series.puiseux_variable",
            message="variable must match the polynomial identifier grammar",
        ) from error
    if (
        not isinstance(window.terms, tuple)
        or len(window.terms) > MAX_LOCAL_SERIES_TERMS
    ):
        _resource(
            "puiseux_terms",
            "Puiseux arithmetic exceeds the retained-term bound",
            ("series", "terms"),
        )
    lower = _check_rational(
        window.valuation_lower, label="valuation_lower", exponent=True
    )
    precision = _check_rational(window.precision, label="precision", exponent=True)
    center = _check_rational(window.center, label="center")
    if lower > precision:
        _domain(
            "puiseux_window",
            "window requires valuation_lower <= precision",
            ("series",),
        )
    if (
        type(window.ramification_index) is not int
        or not 1 <= window.ramification_index <= MAX_PUISEUX_RAMIFICATION
    ):
        _domain(
            "puiseux_ramification",
            "ramification index is outside its domain",
            ("series", "ramification_index"),
        )
    terms: list[tuple[Fraction, Fraction]] = []
    prior: Fraction | None = None
    required = lcm(lower.denominator, precision.denominator)
    if required > MAX_PUISEUX_RAMIFICATION:
        _resource(
            "puiseux_ramification_bound",
            "Puiseux lattice exceeds its ramification bound",
            ("series",),
        )
    for term in window.terms:
        if not isinstance(term, PuiseuxTerm):
            _domain(
                "puiseux_term_type",
                "terms must be PuiseuxTerm values",
                ("series", "terms"),
            )
        exponent = _check_rational(term.exponent, label="exponent", exponent=True)
        coefficient = _check_rational(term.coefficient, label="coefficient")
        if not coefficient:
            _domain(
                "puiseux_zero_term",
                "retained terms must have nonzero coefficients",
                ("series", "terms"),
            )
        if not lower <= exponent < precision:
            _domain(
                "puiseux_term_range",
                "term exponent must lie in the retained window",
                ("series", "terms"),
            )
        if prior is not None and exponent <= prior:
            _domain(
                "puiseux_term_order",
                "terms must have distinct increasing exponents",
                ("series", "terms"),
            )
        prior = exponent
        required = lcm(required, exponent.denominator)
        if required > MAX_PUISEUX_RAMIFICATION:
            _resource(
                "puiseux_ramification_bound",
                "Puiseux lattice exceeds its ramification bound",
                ("series",),
            )
        terms.append((exponent, coefficient))
    if window.ramification_index != required:
        _domain(
            "puiseux_ramification_mismatch",
            "ramification index must be the least common denominator",
            ("series", "ramification_index"),
        )
    return center, lower, tuple(terms), precision


def _pair(
    left: TruncatedPuiseuxWindow, right: TruncatedPuiseuxWindow
) -> tuple[
    tuple[Fraction, Fraction, tuple[tuple[Fraction, Fraction], ...], Fraction],
    tuple[Fraction, Fraction, tuple[tuple[Fraction, Fraction], ...], Fraction],
    int,
]:
    left_data = _check(left)
    right_data = _check(right)
    if (left.variable, left_data[0]) != (right.variable, right_data[0]):
        _domain(
            "puiseux_parent_mismatch",
            "windows must share variable and center",
            ("series",),
        )
    common_ramification = lcm(left.ramification_index, right.ramification_index)
    if common_ramification > MAX_PUISEUX_RAMIFICATION:
        _resource(
            "puiseux_ramification_bound",
            "common Puiseux lattice exceeds its ramification bound",
            ("ramification_index",),
        )
    return left_data, right_data, common_ramification


def _scalar_digits(value: Fraction) -> int:
    return max(abs(value.numerator), value.denominator).bit_length()


def _admit_sum_growth(contributions: list[Fraction]) -> None:
    """Bound exact common-denominator accumulation before constructing sums."""
    if not contributions:
        return
    if len(contributions) == 1:
        value = contributions[0]
        if (
            max(value.numerator.bit_length(), value.denominator.bit_length())
            > _MAX_SCALAR_BITS
        ):
            _resource(
                "puiseux_coefficient_growth",
                "copied coefficient exceeds the scalar digit bound",
                ("terms",),
            )
        return
    denominator_bits = sum(value.denominator.bit_length() for value in contributions)
    numerator_bits = max(value.numerator.bit_length() for value in contributions)
    numerator_bound = numerator_bits + denominator_bits + ceil(log2(len(contributions)))
    if max(denominator_bits, numerator_bound) > _MAX_SCALAR_BITS:
        _resource(
            "puiseux_coefficient_growth",
            "worst-case exact coefficient growth exceeds the scalar digit bound",
            ("terms",),
        )


def _admit_product_sum_growth(
    contributions: list[tuple[Fraction, Fraction]],
) -> None:
    """Bound a sum of products before multiplying any coefficient pair."""
    if not contributions:
        return
    if len(contributions) == 1:
        left, right = contributions[0]
        numerator_bound = left.numerator.bit_length() + right.numerator.bit_length()
        denominator_bound = (
            left.denominator.bit_length() + right.denominator.bit_length()
        )
        if max(numerator_bound, denominator_bound) > _MAX_SCALAR_BITS:
            _resource(
                "puiseux_coefficient_growth",
                "single exact product coefficient growth exceeds the scalar digit bound",
                ("terms",),
            )
        return
    numerator_bits = [
        left.numerator.bit_length() + right.numerator.bit_length()
        for left, right in contributions
    ]
    denominator_bits = [
        left.denominator.bit_length() + right.denominator.bit_length()
        for left, right in contributions
    ]
    denominator_total = sum(denominator_bits)
    numerator_bound = (
        max(numerator_bits) + denominator_total + ceil(log2(len(contributions)))
    )
    if max(denominator_total, numerator_bound) > _MAX_SCALAR_BITS:
        _resource(
            "puiseux_coefficient_growth",
            "worst-case exact product coefficient growth exceeds the scalar digit bound",
            ("terms",),
        )


def _make(
    source: TruncatedPuiseuxWindow,
    lower: Fraction,
    precision: Fraction,
    coefficients: dict[Fraction, Fraction],
    ramification: int,
) -> TruncatedPuiseuxWindow:
    if lower > precision:
        lower = precision
    if (
        abs(lower) > MAX_LOCAL_SERIES_EXPONENT
        or abs(precision) > MAX_LOCAL_SERIES_EXPONENT
    ):
        _resource(
            "puiseux_exponent_bound",
            "result exponent window exceeds its bound",
            ("result",),
        )
    retained = sorted(
        (exponent, value)
        for exponent, value in coefficients.items()
        if value and lower <= exponent < precision
    )
    if len(retained) > MAX_LOCAL_SERIES_TERMS:
        _resource(
            "puiseux_terms",
            "result exceeds the retained-term bound",
            ("result", "terms"),
        )
    normalized_ramification = lcm(
        lower.denominator,
        precision.denominator,
        *(exponent.denominator for exponent, _ in retained),
    )
    if normalized_ramification > MAX_PUISEUX_RAMIFICATION:
        _resource(
            "puiseux_ramification_bound",
            "result lattice exceeds its ramification bound",
            ("result",),
        )
    if normalized_ramification > ramification:
        _domain(
            "puiseux_lattice", "result escaped the admitted common lattice", ("result",)
        )
    terms = []
    for exponent, coefficient in retained:
        if (
            len(str(abs(coefficient.numerator))) > MAX_LOCAL_SERIES_COEFFICIENT_DIGITS
            or len(str(coefficient.denominator)) > MAX_LOCAL_SERIES_COEFFICIENT_DIGITS
        ):
            _resource(
                "puiseux_coefficient_growth",
                "result coefficient exceeds its decimal digit bound",
                ("result", "terms"),
            )
        value = _canonical(coefficient)
        if _scalar_digits(coefficient) > _MAX_SCALAR_BITS:
            _resource(
                "puiseux_coefficient_growth",
                "result coefficient exceeds its digit bound",
                ("result", "terms"),
            )
        terms.append(PuiseuxTerm(exponent=_canonical(exponent), coefficient=value))
    return TruncatedPuiseuxWindow.model_construct(
        variable=source.variable,
        center=source.center,
        valuation_lower=_canonical(lower),
        precision=_canonical(precision),
        ramification_index=normalized_ramification,
        terms=tuple(terms),
    )


def add(
    left: TruncatedPuiseuxWindow, right: TruncatedPuiseuxWindow
) -> TruncatedPuiseuxWindow:
    """Add two windows through the common known cutoff."""
    a, b, ramification = _pair(left, right)
    center, lower_a, terms_a, precision_a = a
    _, lower_b, terms_b, precision_b = b
    lower = min(lower_a, lower_b)
    precision = min(precision_a, precision_b)
    _admit_window_bounds(lower, precision)
    grouped: dict[Fraction, list[Fraction]] = defaultdict(list)
    for exponent, coefficient in (*terms_a, *terms_b):
        if lower <= exponent < precision:
            grouped[exponent].append(coefficient)
    for values in grouped.values():
        _admit_sum_growth(values)
    coefficients = {
        exponent: total
        for exponent, values in grouped.items()
        if (total := sum(values, Fraction()))
    }
    _admit_support(len(coefficients))
    _admit_result_envelope(coefficients, left.variable, center)
    return _make(left, lower, precision, coefficients, ramification)


def subtract(
    left: TruncatedPuiseuxWindow, right: TruncatedPuiseuxWindow
) -> TruncatedPuiseuxWindow:
    """Subtract two windows through the common known cutoff."""
    a, b, ramification = _pair(left, right)
    center, lower_a, terms_a, precision_a = a
    _, lower_b, terms_b, precision_b = b
    lower = min(lower_a, lower_b)
    precision = min(precision_a, precision_b)
    _admit_window_bounds(lower, precision)
    grouped: dict[Fraction, list[Fraction]] = defaultdict(list)
    for exponent, coefficient in terms_a:
        if lower <= exponent < precision:
            grouped[exponent].append(coefficient)
    for exponent, coefficient in terms_b:
        if lower <= exponent < precision:
            grouped[exponent].append(-coefficient)
    for values in grouped.values():
        _admit_sum_growth(values)
    coefficients = {
        exponent: total
        for exponent, values in grouped.items()
        if (total := sum(values, Fraction()))
    }
    _admit_support(len(coefficients))
    _admit_result_envelope(coefficients, left.variable, center)
    return _make(left, lower, precision, coefficients, ramification)


def _admit_window_bounds(lower: Fraction, precision: Fraction) -> None:
    if (
        abs(lower) > MAX_LOCAL_SERIES_EXPONENT
        or abs(precision) > MAX_LOCAL_SERIES_EXPONENT
    ):
        _resource(
            "puiseux_exponent_bound",
            "result exponent window exceeds its bound",
            ("result",),
        )


def _admit_support(term_count: int) -> None:
    if term_count > MAX_LOCAL_SERIES_TERMS:
        _resource(
            "puiseux_terms",
            "result support exceeds the retained-term bound",
            ("result", "terms"),
        )


def _admit_result_envelope(
    coefficients: dict[Fraction, Fraction], variable: str, center: Fraction
) -> None:
    # Estimate the result envelope from admitted input widths and bounded
    # sum/product growth instead of charging every term at the worst-case
    # scalar digit bound. Exponents have denominator <= 256 and absolute
    # value <= 1e6; the fixed allowance covers mapping keys, window bounds,
    # and parent data as decimal digits plus structural overhead, while the
    # result's repeated center is charged at its exact admitted widths. The
    # total is capped at the canonical encoded-output transport ceiling.
    total = (
        len(variable.encode("utf-8"))
        + len(format_canonical_integer(center.numerator))
        + len(format_canonical_integer(center.denominator))
        + 512
    )
    for exponent, value in coefficients.items():
        total += (
            len(format_canonical_integer(exponent.numerator))
            + len(format_canonical_integer(exponent.denominator))
            + len(format_canonical_integer(value.numerator))
            + len(format_canonical_integer(value.denominator))
            + 64
        )
        if total > MAX_PUISEUX_RESULT_DIGITS:
            _resource(
                "puiseux_result_envelope",
                "Puiseux result exceeds its admitted digit envelope",
                ("result",),
            )


def multiply(
    left: TruncatedPuiseuxWindow, right: TruncatedPuiseuxWindow
) -> TruncatedPuiseuxWindow:
    """Multiply windows through the greatest precision known from both tails."""
    a, b, ramification = _pair(left, right)
    center, lower_a, terms_a, precision_a = a
    _, lower_b, terms_b, precision_b = b
    valuation_a = min((exponent for exponent, _ in terms_a), default=precision_a)
    valuation_b = min((exponent for exponent, _ in terms_b), default=precision_b)
    lower = lower_a + lower_b
    precision = min(precision_a + valuation_b, precision_b + valuation_a)
    if lower > precision:
        lower = precision
    if (
        abs(lower) > MAX_LOCAL_SERIES_EXPONENT
        or abs(precision) > MAX_LOCAL_SERIES_EXPONENT
    ):
        _resource(
            "puiseux_exponent_bound",
            "product exponent window exceeds its bound",
            ("result",),
        )
    pair_count = len(terms_a) * len(terms_b)
    if pair_count > MAX_PUISEUX_ARITHMETIC_WORK:
        _resource(
            "puiseux_arithmetic_work",
            "Puiseux product exceeds the convolution work bound",
            ("terms",),
        )
    contributions: dict[Fraction, list[tuple[Fraction, Fraction]]] = defaultdict(list)
    for exponent_a, coefficient_a in terms_a:
        for exponent_b, coefficient_b in terms_b:
            exponent = exponent_a + exponent_b
            if lower <= exponent < precision:
                contributions[exponent].append((coefficient_a, coefficient_b))
    for values in contributions.values():
        _admit_product_sum_growth(values)
    coefficients = {
        exponent: total
        for exponent, values in contributions.items()
        if (total := sum((left * right for left, right in values), Fraction()))
    }
    _admit_support(len(coefficients))
    _admit_result_envelope(coefficients, left.variable, center)
    return _make(left, lower, precision, coefficients, ramification)


def derivative(series: TruncatedPuiseuxWindow) -> TruncatedPuiseuxWindow:
    """Differentiate a Puiseux prefix with respect to its local parameter.

    The derivative of an unknown ``O(t^P)`` tail is only known modulo
    ``O(t^(P-1))``. A zero retained prefix therefore stays a zero *prefix*;
    it does not establish that the underlying series is zero.
    """
    center, lower, terms, precision = _check(series)
    output_lower = lower - 1
    output_precision = precision - 1
    _admit_window_bounds(output_lower, output_precision)
    if len(terms) > MAX_PUISEUX_ARITHMETIC_WORK:
        _resource(
            "puiseux_arithmetic_work",
            "Puiseux derivative exceeds the termwise work bound",
            ("terms",),
        )
    _admit_support(len(terms))
    # The rational exponent is the exact multiplier. Bound every generated
    # coefficient and serialized output before doing the multiplications.
    for exponent, coefficient in terms:
        if exponent:
            _admit_product_sum_growth([(coefficient, exponent)])
    coefficients = {
        exponent - 1: coefficient * exponent
        for exponent, coefficient in terms
        if exponent
    }
    _admit_result_envelope(coefficients, series.variable, center)
    return _make(
        series,
        output_lower,
        output_precision,
        coefficients,
        series.ramification_index,
    )


def _inverse_geometry(
    precision: Fraction,
    terms: tuple[tuple[Fraction, Fraction], ...],
    lattice: int,
) -> tuple[Fraction, Fraction, int, list[tuple[int, Fraction]]]:
    valuation = terms[0][0]
    output_lower = -valuation
    output_precision = precision - 2 * valuation
    _admit_window_bounds(output_lower, output_precision)
    lattice_span = (precision - valuation) * lattice
    if lattice_span.denominator != 1:
        _domain(
            "puiseux_inverse_lattice",
            "inverse precision does not lie on the source ramification lattice",
            ("series",),
        )
    coefficient_count = lattice_span.numerator
    if coefficient_count < 1:
        _domain(
            "puiseux_inverse_precision",
            "source precision must exceed the retained valuation",
            ("series", "precision"),
        )

    unit_terms = []
    for exponent, coefficient in terms[1:]:
        lattice_index = (exponent - valuation) * lattice
        if lattice_index.denominator != 1 or lattice_index <= 0:
            _domain(
                "puiseux_inverse_lattice",
                "retained unit terms must lie on the source ramification lattice",
                ("series", "terms"),
            )
        unit_terms.append((lattice_index.numerator, coefficient))
    return output_lower, output_precision, coefficient_count, unit_terms


def _admit_inverse_work(
    coefficient_count: int,
    unit_term_count: int,
) -> None:
    work = (coefficient_count - 1) * unit_term_count
    if work > MAX_PUISEUX_ARITHMETIC_WORK:
        _resource(
            "puiseux_arithmetic_work",
            "Puiseux inverse exceeds the admitted coefficient recurrence work",
            ("series", "terms"),
        )


def _inverse_integer_weights(
    terms: tuple[tuple[Fraction, Fraction], ...],
    leading: Fraction,
    unit_terms: list[tuple[int, Fraction]],
    coefficient_count: int,
) -> tuple[dict[int, int], int]:
    """Normalize exact unit coefficients after bounding each integer growth."""
    source_bits = max(
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        for _, value in terms
    )
    leading_bits = max(
        abs(leading.numerator).bit_length(), leading.denominator.bit_length()
    )
    ratios: list[tuple[int, Fraction]] = []
    for index, coefficient in unit_terms:
        if source_bits + leading_bits > _MAX_SCALAR_BITS:
            _resource(
                "puiseux_coefficient_growth",
                "normalized unit coefficient exceeds the exact scalar digit bound",
                ("series", "terms"),
            )
        ratios.append((index, coefficient / leading))

    common_denominator = 1
    for _, ratio in ratios:
        divisor = gcd(common_denominator, ratio.denominator)
        quotient = common_denominator // divisor
        if quotient.bit_length() + ratio.denominator.bit_length() > _MAX_SCALAR_BITS:
            _resource(
                "puiseux_coefficient_growth",
                "common unit denominator exceeds the exact scalar digit bound",
                ("series", "terms"),
            )
        common_denominator = quotient * ratio.denominator

    integer_coefficients = []
    for index, ratio in ratios:
        scaled_denominator = common_denominator // ratio.denominator
        if (
            abs(ratio.numerator).bit_length() + scaled_denominator.bit_length()
            > _MAX_SCALAR_BITS
        ):
            _resource(
                "puiseux_coefficient_growth",
                "integer unit coefficient exceeds the exact scalar digit bound",
                ("series", "terms"),
            )
        integer_coefficients.append((index, ratio.numerator * scaled_denominator))

    denominator_bits = common_denominator.bit_length()
    maximum_weight_bits = max(
        coefficient.bit_length() + (index - 1) * denominator_bits
        for index, coefficient in integer_coefficients
    )
    summand_count_bits = (len(integer_coefficients) - 1).bit_length()
    recurrence_bits = 1 + (coefficient_count - 1) * (
        maximum_weight_bits + summand_count_bits
    )
    numerator_bound = recurrence_bits + leading.denominator.bit_length()
    denominator_bound = (coefficient_count - 1) * denominator_bits + abs(
        leading.numerator
    ).bit_length()
    if max(numerator_bound, denominator_bound) > _MAX_SCALAR_BITS:
        _resource(
            "puiseux_coefficient_growth",
            "inverse recurrence may exceed the exact scalar digit bound",
            ("result", "terms"),
        )

    weights: dict[int, int] = {}
    for lattice_index, integer_value in integer_coefficients:
        weights[lattice_index] = integer_value * common_denominator ** (
            lattice_index - 1
        )
    return weights, common_denominator


def _inverse_unit_coefficients(
    coefficient_count: int,
    weights: dict[int, int],
    denominator: int,
    leading: Fraction,
    lower: Fraction,
    lattice: int,
) -> dict[Fraction, Fraction]:
    unit_inverse = [1]
    for index in range(1, coefficient_count):
        unit_inverse.append(
            -sum(
                weight * unit_inverse[index - shift]
                for shift, weight in weights.items()
                if shift <= index
            )
        )

    coefficients: dict[Fraction, Fraction] = {}
    denominator_power = 1
    for index, value in enumerate(unit_inverse):
        if value:
            exponent = lower + Fraction(index, lattice)
            coefficients[exponent] = Fraction(value, denominator_power) / leading
        denominator_power *= denominator
    return coefficients


def inverse(series: TruncatedPuiseuxWindow) -> TruncatedPuiseuxWindow:
    """Invert a Puiseux prefix when its nonzero leading term is retained.

    If the known prefix has valuation ``v`` and ends at precision ``P``, the
    inverse is known through ``O(t^(P-2v))``. An empty prefix cannot establish
    the valuation of its unknown tail and is therefore not invertible from
    the supplied information.
    """
    center, _, terms, source_precision = _check(series)
    if not terms:
        _domain(
            "puiseux_inverse_undetermined",
            "inverse requires a retained nonzero leading term; an empty finite prefix may have an unknown nonzero tail",
            ("series", "terms"),
        )

    _, leading = terms[0]
    lower, precision, count, unit_terms = _inverse_geometry(
        source_precision, terms, series.ramification_index
    )
    _admit_inverse_work(count, len(unit_terms))
    if not unit_terms:
        reciprocal = Fraction(1, 1) / leading
        if (
            max(reciprocal.numerator.bit_length(), reciprocal.denominator.bit_length())
            > _MAX_SCALAR_BITS
        ):
            _resource(
                "puiseux_coefficient_growth",
                "inverse coefficient exceeds the exact scalar digit bound",
                ("result", "terms"),
            )
        coefficients = {lower: reciprocal}
    else:
        weights, denominator = _inverse_integer_weights(
            terms, leading, unit_terms, count
        )
        coefficients = _inverse_unit_coefficients(
            count,
            weights,
            denominator,
            leading,
            lower,
            series.ramification_index,
        )
    _admit_support(len(coefficients))
    _admit_result_envelope(coefficients, series.variable, center)
    return _make(series, lower, precision, coefficients, series.ramification_index)


__all__ = [
    "MAX_PUISEUX_ARITHMETIC_WORK",
    "MAX_PUISEUX_RESULT_DIGITS",
    "add",
    "derivative",
    "inverse",
    "multiply",
    "subtract",
]
