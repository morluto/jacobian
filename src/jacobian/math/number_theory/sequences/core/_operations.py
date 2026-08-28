"""Exact integer-sequence operations using the Python standard library."""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from functools import reduce
from itertools import pairwise

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.sequences.core._models import (
    FrequencyEntry,
    IntegerSequenceBooleanResult,
    IntegerSequenceFrequenciesResult,
    IntegerSequenceIndexListResult,
    IntegerSequenceListResult,
    IntegerSequenceRationalResult,
    IntegerSequenceRequest,
    IntegerSequenceValueResult,
)
from jacobian.math.number_theory.sequences.core.values import (
    MAX_SEQUENCE_WIRE_BYTES,
    IntegerSequence,
)


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
    estimated = output_items * (output_digits + 3) + 64
    if output_items and estimated > MAX_SEQUENCE_WIRE_BYTES:
        raise OperationDomainValidationError(
            location=("values",),
            code="sequences.result_transport_too_large",
            message=(
                "the exact result exceeds the "
                f"{MAX_SEQUENCE_WIRE_BYTES}-byte transport envelope"
            ),
        )
    return values


def _digits(values: tuple[str, ...]) -> int:
    """Return a decimal-width bound from canonical source strings."""

    return max((len(value.lstrip("-")) for value in values), default=1)


def _multiplicative_digits(request: IntegerSequence, *, prefix: bool = False) -> int:
    """Bound product-like widths, accounting for the absorbing zero value."""

    total = 0
    maximum = 1
    for value in request.values:
        if value == "0":
            return maximum if prefix else 1
        total += len(value.lstrip("-"))
        maximum = max(maximum, total)
    return maximum


def _values(request: IntegerSequenceRequest) -> list[int]:
    return [parse_canonical_integer(value) for value in request.values]


def _value_result(value: int) -> IntegerSequenceValueResult:
    return IntegerSequenceValueResult(value=format_canonical_integer(value))


def _list_result(values: list[int]) -> IntegerSequenceListResult:
    return IntegerSequenceListResult(
        values=tuple(format_canonical_integer(value) for value in values)
    )


def sequence_sum(request: IntegerSequenceRequest) -> IntegerSequenceValueResult:
    values = _admit(
        request, output_digits=_digits(request.values) + len(str(len(request.values)))
    )
    return _value_result(sum(values))


def sequence_product(request: IntegerSequenceRequest) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_multiplicative_digits(request))
    return _value_result(math.prod(values))


def sequence_gcd(request: IntegerSequenceRequest) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_digits(request.values))
    return _value_result(reduce(math.gcd, values, 0))


def sequence_lcm(request: IntegerSequenceRequest) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_multiplicative_digits(request))
    return _value_result(reduce(math.lcm, values, 1))


def sequence_minimum(request: IntegerSequenceRequest) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_digits(request.values))
    return _value_result(min(values))


def sequence_maximum(request: IntegerSequenceRequest) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_digits(request.values))
    return _value_result(max(values))


def sequence_range(request: IntegerSequenceRequest) -> IntegerSequenceValueResult:
    values = _admit(request, output_digits=_digits(request.values) + 1)
    return _value_result(max(values) - min(values))


def sequence_mean(request: IntegerSequenceRequest) -> IntegerSequenceRationalResult:
    values = _admit(
        request, output_digits=_digits(request.values) + len(str(len(request.values)))
    )
    fraction = Fraction(sum(values), len(values))
    return IntegerSequenceRationalResult(
        value=CanonicalRational(
            num=format_canonical_integer(fraction.numerator),
            den=format_canonical_integer(fraction.denominator),
        )
    )


def sequence_median(request: IntegerSequenceRequest) -> IntegerSequenceRationalResult:
    values = sorted(_admit(request, output_digits=_digits(request.values) + 1))
    middle = len(values) // 2
    if len(values) % 2:
        fraction = Fraction(values[middle])
    else:
        fraction = Fraction(values[middle - 1] + values[middle], 2)
    return IntegerSequenceRationalResult(
        value=CanonicalRational(
            num=format_canonical_integer(fraction.numerator),
            den=format_canonical_integer(fraction.denominator),
        )
    )


def sequence_distinct_count(
    request: IntegerSequenceRequest,
) -> IntegerSequenceValueResult:
    _admit(request, output_digits=len(str(len(request.values))))
    return _value_result(len(set(_values(request))))


def prefix_sums(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
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


def first_differences(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values) + 1,
        output_items=max(len(request.values) - 1, 0),
    )
    return _list_result([right - left for left, right in pairwise(values)])


def second_differences(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values) + 2,
        output_items=max(len(request.values) - 2, 0),
    )
    first = [right - left for left, right in pairwise(values)]
    return _list_result([right - left for left, right in pairwise(first)])


def prefix_products(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
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


def prefix_minima(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    result = [values[0]]
    for value in values[1:]:
        result.append(min(result[-1], value))
    return _list_result(result)


def prefix_maxima(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    result = [values[0]]
    for value in values[1:]:
        result.append(max(result[-1], value))
    return _list_result(result)


def prefix_gcds(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    result = [abs(values[0])]
    for value in values[1:]:
        result.append(math.gcd(result[-1], value))
    return _list_result(result)


def prefix_lcms(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_multiplicative_digits(request, prefix=True),
        output_items=len(request.values),
    )
    result = [abs(values[0])]
    for value in values[1:]:
        result.append(math.lcm(result[-1], value))
    return _list_result(result)


def sorted_unique(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    return _list_result(sorted(set(values)))


def sort_sequence(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    return _list_result(sorted(values))


def reverse_sequence(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    return _list_result(list(reversed(values)))


def parities(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _admit(request, output_digits=1, output_items=len(request.values))
    return _list_result([value % 2 for value in values])


def signs(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _admit(request, output_digits=1, output_items=len(request.values))
    return _list_result([(value > 0) - (value < 0) for value in values])


def frequencies(request: IntegerSequenceRequest) -> IntegerSequenceFrequenciesResult:
    values = _admit(
        request,
        output_digits=_digits(request.values),
        output_items=len(request.values),
    )
    counts = Counter(values)
    entries = tuple(
        FrequencyEntry(value=format_canonical_integer(value), count=counts[value])
        for value in sorted(counts)
    )
    return IntegerSequenceFrequenciesResult(entries=entries)


def zero_indices(request: IntegerSequenceRequest) -> IntegerSequenceIndexListResult:
    values = _admit(
        request,
        output_digits=len(str(len(request.values))),
        output_items=len(request.values),
    )
    return IntegerSequenceIndexListResult(
        indices=tuple(index for index, value in enumerate(values) if value == 0),
    )


def decide_arithmetic(request: IntegerSequenceRequest) -> IntegerSequenceBooleanResult:
    values = _admit(request, output_digits=1)
    if len(values) < 2:
        return IntegerSequenceBooleanResult(holds=True)
    differences = {right - left for left, right in pairwise(values)}
    return IntegerSequenceBooleanResult(holds=len(differences) <= 1)


def decide_geometric(request: IntegerSequenceRequest) -> IntegerSequenceBooleanResult:
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
    request: IntegerSequenceRequest,
) -> IntegerSequenceBooleanResult:
    values = _admit(request, output_digits=1)
    return IntegerSequenceBooleanResult(
        holds=all(left <= right for left, right in pairwise(values))
    )


def decide_strictly_increasing(
    request: IntegerSequenceRequest,
) -> IntegerSequenceBooleanResult:
    values = _admit(request, output_digits=1)
    return IntegerSequenceBooleanResult(
        holds=all(left < right for left, right in pairwise(values))
    )
