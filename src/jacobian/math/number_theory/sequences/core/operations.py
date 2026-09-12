"""Native exact operations on finite integer and rational sequences."""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from functools import reduce
from itertools import pairwise

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
)
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.sequences.core._models import (
    AutocorrelationCell,
    AutocorrelationResult,
    FiniteIntegerSequence,
    FiniteRationalSequence,
    FrequencyEntry,
    IntegerSequenceBooleanResult,
    IntegerSequenceFrequenciesResult,
    IntegerSequenceIndexListResult,
    IntegerSequenceListResult,
    IntegerSequenceRationalResult,
    IntegerSequenceValueResult,
    SequenceLogConcavityRow,
    SequenceOrderShapeResult,
)
from jacobian.math.number_theory.sequences.core.values import (
    MAX_SEQUENCE_TOTAL_DIGITS,
    IntegerSequence,
)

MAX_AUTOCORRELATION_MULTIPLICATIONS = 4_000_000
MAX_ORDER_SHAPE_WORK = 4_000_000
MAX_ORDER_SHAPE_RESULT_ALLOCATIONS = 500_000


def _admit(
    request: IntegerSequence,
    *,
    output_digits: int,
    output_items: int = 1,
) -> list[int]:
    """Admit one operation's complete source/result envelope.

    The reusable sequence value checks only its canonical representation.  A
    native operation owns the derived output budget because sums, products,
    and materialized transforms have different growth envelopes.
    """

    values = _values(request)
    if output_items and output_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("values",),
            code="sequences.result_digits_exceeded",
            message=(
                "the exact result exceeds the "
                f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit bound"
            ),
        )
    total_output_digits = output_items * output_digits
    if output_items and total_output_digits > MAX_SEQUENCE_TOTAL_DIGITS:
        raise OperationDomainValidationError(
            location=("values",),
            code="sequences.result_representation_too_large",
            message=(
                "the exact result exceeds the "
                f"{MAX_SEQUENCE_TOTAL_DIGITS}-digit representation bound"
            ),
        )
    return values


def _digits(values: tuple[int, ...]) -> int:
    """Return a decimal-width bound from native source integers."""

    return max(
        (len(format_canonical_integer(abs(value))) for value in values), default=1
    )


def _multiplicative_digits(request: IntegerSequence, *, prefix: bool = False) -> int:
    """Bound product-like widths, accounting for the absorbing zero value."""

    total = 0
    maximum = 1
    for value in request.values:
        if value == 0:
            return maximum if prefix else 1
        total += len(format_canonical_integer(abs(value)))
        maximum = max(maximum, total)
    return maximum


def _values(request: IntegerSequence) -> list[int]:
    return list(request.values)


def _value_result(value: int) -> IntegerSequenceValueResult:
    return IntegerSequenceValueResult(value=value)


def _admit_autocorrelation(request: FiniteIntegerSequence) -> tuple[int, ...]:
    values = tuple(request.values)
    if not values:
        return values
    digits = max(len(format_canonical_integer(abs(value))) for value in values)
    result_digits = 2 * digits + len(str(len(values)))
    if result_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("values",),
            code="sequences.autocorrelation_result_digits_exceeded",
            message="autocorrelation values exceed the exact integer digit bound",
        )
    return values


def _order_shape_rationals(
    request: FiniteRationalSequence,
) -> tuple[CanonicalRational, ...]:
    if isinstance(request, FiniteRationalSequence):
        return request.values
    raise TypeError("sequence_order_shape requires FiniteRationalSequence")


def _admit_order_shape(
    request: FiniteRationalSequence,
) -> tuple[CanonicalRational, ...]:
    """Admit linear comparisons, exact products, and the complete profile."""

    rational_values = _order_shape_rationals(request)
    size = len(rational_values)
    component_digits = max(
        (
            max(
                len(format_canonical_integer(abs(value.num))),
                len(format_canonical_integer(value.den)),
            )
            for value in rational_values
        ),
        default=1,
    )
    source_digits = sum(
        len(format_canonical_integer(abs(value.num)))
        + len(format_canonical_integer(value.den))
        for value in rational_values
    )
    if size * component_digits > MAX_ORDER_SHAPE_WORK:
        raise OperationResourceAdmissionError(
            location=("values",),
            code="sequences.order_shape.work_bound",
            message="order-shape comparisons exceed the admitted exact-work bound",
        )
    row_count = max(0, size - 2)
    product_digits = 2 * component_digits
    if row_count and product_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("values",),
            code="sequences.order_shape.result_digits_exceeded",
            message="order-shape cross-products exceed the exact rational digit bound",
        )
    # The result retains the source, one peak-position slot per source entry,
    # and four slots (index, two exact products, decision) per interior row.
    result_allocations = size + size + 4 * row_count + 8
    if result_allocations > MAX_ORDER_SHAPE_RESULT_ALLOCATIONS:
        raise OperationResourceAdmissionError(
            location=("values",),
            code="sequences.order_shape.result_allocation_bound",
            message=(
                "the complete order-shape profile requires "
                f"{result_allocations} result allocations; maximum is "
                f"{MAX_ORDER_SHAPE_RESULT_ALLOCATIONS}"
            ),
        )
    # Interior rows retain two rationals. Each product can have up to
    # ``product_digits`` digits in both its numerator and denominator, and the
    # result also retains the complete rational source. Empty, singleton, and
    # adjacent pairs construct no product rows.
    result_digits = source_digits + row_count * 4 * product_digits
    if result_digits > MAX_SEQUENCE_TOTAL_DIGITS:
        raise OperationDomainValidationError(
            location=("values",),
            code="sequences.order_shape.result_representation_too_large",
            message=(
                "order-shape cross-products exceed the exact result representation "
                f"bound of {MAX_SEQUENCE_TOTAL_DIGITS} digits"
            ),
        )
    return rational_values


def aperiodic_autocorrelation(request: FiniteIntegerSequence) -> AutocorrelationResult:
    values = _admit_autocorrelation(request)
    size = len(values)
    if size * (size + 1) // 2 > MAX_AUTOCORRELATION_MULTIPLICATIONS:
        raise OperationResourceAdmissionError(
            location=("values",),
            code="sequences.autocorrelation.work_bound",
            message="aperiodic autocorrelation exceeds the admitted multiplication bound",
        )
    cells = tuple(
        AutocorrelationCell(
            lag=lag,
            value=sum(
                values[index] * values[index + lag] for index in range(size - lag)
            ),
        )
        for lag in range(size)
    )
    negative = tuple(
        AutocorrelationCell(lag=-cell.lag, value=cell.value)
        for cell in reversed(cells[1:])
    )
    return AutocorrelationResult(source=request, cells=negative + cells)


def cyclic_autocorrelation(request: FiniteIntegerSequence) -> AutocorrelationResult:
    values = _admit_autocorrelation(request)
    size = len(values)
    if size * size > MAX_AUTOCORRELATION_MULTIPLICATIONS:
        raise OperationResourceAdmissionError(
            location=("values",),
            code="sequences.autocorrelation.work_bound",
            message="cyclic autocorrelation exceeds the admitted multiplication bound",
        )
    return AutocorrelationResult(
        source=request,
        cells=tuple(
            AutocorrelationCell(
                lag=lag,
                value=sum(
                    values[index] * values[(index + lag) % size]
                    for index in range(size)
                ),
            )
            for lag in range(size)
        ),
    )


def _order_shape_scalar(
    value: Fraction,
) -> CanonicalRational:
    """Encode a product in the canonical rational result domain."""

    return CanonicalRational.from_fraction(value)


def sequence_order_shape(
    request: FiniteRationalSequence,
) -> SequenceOrderShapeResult:
    fractions = tuple(value.as_fraction() for value in _admit_order_shape(request))
    nondecreasing_violation = next(
        (
            index
            for index in range(len(fractions) - 1)
            if fractions[index] > fractions[index + 1]
        ),
        None,
    )
    nonincreasing_violation = next(
        (
            index
            for index in range(len(fractions) - 1)
            if fractions[index] < fractions[index + 1]
        ),
        None,
    )
    nondecreasing_prefix = [True] * len(fractions)
    for index in range(1, len(fractions)):
        nondecreasing_prefix[index] = (
            nondecreasing_prefix[index - 1] and fractions[index - 1] <= fractions[index]
        )
    nonincreasing_suffix = [True] * len(fractions)
    for index in range(len(fractions) - 2, -1, -1):
        nonincreasing_suffix[index] = (
            nonincreasing_suffix[index + 1] and fractions[index] >= fractions[index + 1]
        )
    peaks = tuple(
        index
        for index in range(len(fractions))
        if nondecreasing_prefix[index] and nonincreasing_suffix[index]
    )
    log_rows = tuple(
        SequenceLogConcavityRow(
            index=index,
            square=_order_shape_scalar(fractions[index] ** 2),
            neighbor_product=_order_shape_scalar(
                fractions[index - 1] * fractions[index + 1]
            ),
            holds=fractions[index] ** 2 >= fractions[index - 1] * fractions[index + 1],
        )
        for index in range(1, len(fractions) - 1)
    )
    first_log_violation = next((row.index for row in log_rows if not row.holds), None)
    first_negative = next(
        (index for index, value in enumerate(fractions) if value < 0), None
    )
    nonzero = [index for index, value in enumerate(fractions) if value != 0]
    internal_zero_indices = (
        tuple(
            index
            for index in range(nonzero[0] + 1, nonzero[-1])
            if fractions[index] == 0
        )
        if nonzero
        else ()
    )
    first_internal_zero = internal_zero_indices[0] if internal_zero_indices else None
    return SequenceOrderShapeResult(
        source=request,
        first_nondecreasing_violation=nondecreasing_violation,
        first_nonincreasing_violation=nonincreasing_violation,
        weak_unimodal_peak_positions=peaks,
        log_concavity_rows=log_rows,
        first_log_concavity_violation=first_log_violation,
        is_nonnegative=first_negative is None,
        first_negative_index=first_negative,
        has_internal_zero=bool(internal_zero_indices),
        first_internal_zero_index=first_internal_zero,
    )


def _list_result(values: list[int]) -> IntegerSequenceListResult:
    return IntegerSequenceListResult(values=tuple(values))


def sequence_sum(request: IntegerSequence) -> IntegerSequenceValueResult:
    values = _admit(
        request, output_digits=_digits(request.values) + len(str(len(request.values)))
    )
    return _value_result(sum(values))


def sequence_product(request: IntegerSequence) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_multiplicative_digits(request))
    return _value_result(math.prod(values))


def sequence_gcd(request: IntegerSequence) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_digits(request.values))
    return _value_result(reduce(math.gcd, values, 0))


def sequence_lcm(request: IntegerSequence) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_multiplicative_digits(request))
    return _value_result(reduce(math.lcm, values, 1))


def sequence_minimum(request: IntegerSequence) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_digits(request.values))
    return _value_result(min(values))


def sequence_maximum(request: IntegerSequence) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_digits(request.values))
    return _value_result(max(values))


def sequence_range(request: IntegerSequence) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_digits(request.values) + 1)
    return _value_result(max(values) - min(values))


def sequence_mean(request: IntegerSequence) -> IntegerSequenceRationalResult:
    values = _admit(
        request, output_digits=_digits(request.values) + len(str(len(request.values)))
    )
    fraction = Fraction(sum(values), len(values))
    return IntegerSequenceRationalResult(
        value=CanonicalRational(
            num=fraction.numerator,
            den=fraction.denominator,
        )
    )


def sequence_median(request: IntegerSequence) -> IntegerSequenceRationalResult:
    values = sorted(_admit(request, output_digits=_digits(request.values) + 1))
    middle = len(values) // 2
    if len(values) % 2:
        fraction = Fraction(values[middle])
    else:
        fraction = Fraction(values[middle - 1] + values[middle], 2)
    return IntegerSequenceRationalResult(
        value=CanonicalRational(
            num=fraction.numerator,
            den=fraction.denominator,
        )
    )


def sequence_distinct_count(
    request: IntegerSequence,
) -> IntegerSequenceValueResult:
    _admit(request, output_digits=len(str(len(request.values))))
    return _value_result(len(set(_values(request))))


def prefix_sums(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values) + len(str(len(request.values))),
        output_items=len(request.values),
    )
    total = 0
    result: list[int] = []
    for value in values:
        total += value
        result.append(total)
    return _list_result(result)


def first_differences(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values) + 1,
        output_items=max(len(request.values) - 1, 0),
    )
    return _list_result([right - left for left, right in pairwise(values)])


def second_differences(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values) + 2,
        output_items=max(len(request.values) - 2, 0),
    )
    first = [right - left for left, right in pairwise(values)]
    return _list_result([right - left for left, right in pairwise(first)])


def prefix_products(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_multiplicative_digits(request, prefix=True),
        output_items=len(request.values),
    )
    total = 1
    result: list[int] = []
    for value in values:
        total *= value
        result.append(total)
    return _list_result(result)


def prefix_minima(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    result = [values[0]]
    for value in values[1:]:
        result.append(min(result[-1], value))
    return _list_result(result)


def prefix_maxima(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    result = [values[0]]
    for value in values[1:]:
        result.append(max(result[-1], value))
    return _list_result(result)


def prefix_gcds(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    result = [abs(values[0])]
    for value in values[1:]:
        result.append(math.gcd(result[-1], value))
    return _list_result(result)


def prefix_lcms(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_multiplicative_digits(request, prefix=True),
        output_items=len(request.values),
    )
    result = [abs(values[0])]
    for value in values[1:]:
        result.append(math.lcm(result[-1], value))
    return _list_result(result)


def sorted_unique(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    return _list_result(sorted(set(values)))


def sort_sequence(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    return _list_result(sorted(values))


def reverse_sequence(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    return _list_result(list(reversed(values)))


def parities(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(request, output_digits=1, output_items=len(request.values))
    return _list_result([value % 2 for value in values])


def signs(request: IntegerSequence) -> IntegerSequenceListResult:
    values = _admit(request, output_digits=1, output_items=len(request.values))
    return _list_result([(value > 0) - (value < 0) for value in values])


def frequencies(request: IntegerSequence) -> IntegerSequenceFrequenciesResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    counts = Counter(values)
    entries = tuple(
        FrequencyEntry(value=value, count=counts[value]) for value in sorted(counts)
    )
    return IntegerSequenceFrequenciesResult(entries=entries)


def zero_indices(request: IntegerSequence) -> IntegerSequenceIndexListResult:
    values = _admit(
        request,
        output_digits=len(str(len(request.values))),
        output_items=len(request.values),
    )
    return IntegerSequenceIndexListResult(
        indices=tuple(index for index, value in enumerate(values) if value == 0),
    )


def decide_arithmetic(request: IntegerSequence) -> IntegerSequenceBooleanResult:
    values = _admit(request, output_digits=1)
    if len(values) < 2:
        return IntegerSequenceBooleanResult(holds=True)
    differences = {right - left for left, right in pairwise(values)}
    return IntegerSequenceBooleanResult(holds=len(differences) <= 1)


def decide_geometric(request: IntegerSequence) -> IntegerSequenceBooleanResult:
    values = _admit(request, output_digits=1)
    if len(values) < 2:
        return IntegerSequenceBooleanResult(holds=True)
    if values[0] == 0:
        return IntegerSequenceBooleanResult(holds=all(value == 0 for value in values))
    ratio = Fraction(values[1], values[0])
    return IntegerSequenceBooleanResult(
        holds=all(
            right * ratio.denominator == left * ratio.numerator
            for left, right in pairwise(values)
        )
    )


def decide_nondecreasing(
    request: IntegerSequence,
) -> IntegerSequenceBooleanResult:
    values = _admit(request, output_digits=1)
    return IntegerSequenceBooleanResult(
        holds=all(left <= right for left, right in pairwise(values))
    )


def decide_strictly_increasing(
    request: IntegerSequence,
) -> IntegerSequenceBooleanResult:
    values = _admit(request, output_digits=1)
    return IntegerSequenceBooleanResult(
        holds=all(left < right for left, right in pairwise(values))
    )


__all__ = [
    "decide_arithmetic",
    "decide_geometric",
    "decide_nondecreasing",
    "decide_strictly_increasing",
    "first_differences",
    "frequencies",
    "parities",
    "prefix_gcds",
    "prefix_lcms",
    "prefix_maxima",
    "prefix_minima",
    "prefix_products",
    "prefix_sums",
    "reverse_sequence",
    "second_differences",
    "sequence_distinct_count",
    "sequence_gcd",
    "sequence_lcm",
    "sequence_maximum",
    "sequence_mean",
    "sequence_median",
    "sequence_minimum",
    "sequence_product",
    "sequence_range",
    "sequence_sum",
    "signs",
    "sort_sequence",
    "sorted_unique",
    "zero_indices",
]
