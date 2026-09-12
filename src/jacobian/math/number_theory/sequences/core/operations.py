"""Native exact operations on finite integer sequences."""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from functools import reduce
from itertools import pairwise

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import encode_strict_json, format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.sequences.core._models import (
    AutocorrelationCell,
    AutocorrelationResult,
    FiniteIntegerSequence,
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
MAX_SEQUENCE_ORDER_SHAPE_RESULT_BYTES = 10 * 1024 * 1024
_ORDER_SHAPE_ROW_OVERHEAD_BYTES = 160


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


def _admit_order_shape(request: FiniteIntegerSequence) -> tuple[int, ...]:
    """Admit all exact cross-products and the complete profile envelope."""

    values = _admit_autocorrelation(request)
    size = len(values)
    digits = max(
        (len(format_canonical_integer(abs(value))) for value in values), default=1
    )
    cross_product_digits = 2 * digits
    source_bytes = len(encode_strict_json(request.model_dump(mode="json")))
    # Each interior row contains two exact products plus its index and boolean;
    # the fixed allowance covers keys, punctuation, and integer encodings.
    predicted_result_bytes = (
        source_bytes
        + max(0, size - 2)
        * (_ORDER_SHAPE_ROW_OVERHEAD_BYTES + 2 * cross_product_digits)
        + size * 8  # peak-position list and scalar metadata
    )
    if predicted_result_bytes > MAX_SEQUENCE_ORDER_SHAPE_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("values",),
            code="sequences.order_shape.result_bytes_bound",
            message=(
                "the complete order-shape profile is predicted to occupy "
                f"{predicted_result_bytes} bytes; maximum is "
                f"{MAX_SEQUENCE_ORDER_SHAPE_RESULT_BYTES}"
            ),
        )
    return values


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


def sequence_order_shape(request: FiniteIntegerSequence) -> SequenceOrderShapeResult:
    values = _admit_order_shape(request)
    nondecreasing_violation = next(
        (
            index
            for index in range(len(values) - 1)
            if values[index] > values[index + 1]
        ),
        None,
    )
    nonincreasing_violation = next(
        (
            index
            for index in range(len(values) - 1)
            if values[index] < values[index + 1]
        ),
        None,
    )
    nondecreasing_prefix = [True] * len(values)
    for index in range(1, len(values)):
        nondecreasing_prefix[index] = (
            nondecreasing_prefix[index - 1] and values[index - 1] <= values[index]
        )
    nonincreasing_suffix = [True] * len(values)
    for index in range(len(values) - 2, -1, -1):
        nonincreasing_suffix[index] = (
            nonincreasing_suffix[index + 1] and values[index] >= values[index + 1]
        )
    peaks = tuple(
        index
        for index in range(len(values))
        if nondecreasing_prefix[index] and nonincreasing_suffix[index]
    )
    log_rows = tuple(
        SequenceLogConcavityRow(
            index=index,
            square=values[index] ** 2,
            neighbor_product=values[index - 1] * values[index + 1],
            holds=values[index] ** 2 >= values[index - 1] * values[index + 1],
        )
        for index in range(1, len(values) - 1)
    )
    nonzero = [index for index, value in enumerate(values) if value != 0]
    has_internal_zero = bool(nonzero) and any(
        values[index] == 0 for index in range(nonzero[0] + 1, nonzero[-1])
    )
    return SequenceOrderShapeResult(
        source=request,
        first_nondecreasing_violation=nondecreasing_violation,
        first_nonincreasing_violation=nonincreasing_violation,
        weak_unimodal_peak_positions=peaks,
        log_concavity_rows=log_rows,
        is_nonnegative=all(value >= 0 for value in values),
        has_internal_zero=has_internal_zero,
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
