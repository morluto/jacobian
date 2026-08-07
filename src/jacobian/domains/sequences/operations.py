"""Exact integer-sequence operations using the Python standard library."""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from functools import reduce
from itertools import pairwise

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.sequences import (
    FrequencyEntry,
    IntegerSequenceBooleanResult,
    IntegerSequenceFrequenciesResult,
    IntegerSequenceIndexListResult,
    IntegerSequenceListResult,
    IntegerSequenceRationalResult,
    IntegerSequenceRequest,
    IntegerSequenceValueResult,
)


def _values(request: IntegerSequenceRequest) -> list[int]:
    return [int(value) for value in request.values]


def _value_result(value: int) -> IntegerSequenceValueResult:
    return IntegerSequenceValueResult(value=str(value))


def _list_result(values: list[int]) -> IntegerSequenceListResult:
    return IntegerSequenceListResult(values=tuple(str(v) for v in values))


def sequence_sum(request: IntegerSequenceRequest) -> IntegerSequenceValueResult:
    return _value_result(sum(_values(request)))


def sequence_product(request: IntegerSequenceRequest) -> IntegerSequenceValueResult:
    return _value_result(math.prod(_values(request)))


def sequence_gcd(request: IntegerSequenceRequest) -> IntegerSequenceValueResult:
    return _value_result(reduce(math.gcd, _values(request)))


def sequence_lcm(request: IntegerSequenceRequest) -> IntegerSequenceValueResult:
    return _value_result(reduce(math.lcm, _values(request), 1))


def sequence_minimum(request: IntegerSequenceRequest) -> IntegerSequenceValueResult:
    return _value_result(min(_values(request)))


def sequence_maximum(request: IntegerSequenceRequest) -> IntegerSequenceValueResult:
    return _value_result(max(_values(request)))


def sequence_range(request: IntegerSequenceRequest) -> IntegerSequenceValueResult:
    values = _values(request)
    return _value_result(max(values) - min(values))


def sequence_mean(request: IntegerSequenceRequest) -> IntegerSequenceRationalResult:
    values = _values(request)
    fraction = Fraction(sum(values), len(values))
    return IntegerSequenceRationalResult(
        value=CanonicalRational(
            num=format_canonical_integer(fraction.numerator),
            den=format_canonical_integer(fraction.denominator),
        )
    )


def sequence_median(request: IntegerSequenceRequest) -> IntegerSequenceRationalResult:
    values = sorted(_values(request))
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
    return _value_result(len(set(_values(request))))


def prefix_sums(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    total = 0
    result: list[int] = []
    for value in _values(request):
        total += value
        result.append(total)
    return _list_result(result)


def first_differences(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _values(request)
    return _list_result([right - left for left, right in pairwise(values)])


def second_differences(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _values(request)
    first = [right - left for left, right in pairwise(values)]
    return _list_result([right - left for left, right in pairwise(first)])


def prefix_products(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    total = 1
    result: list[int] = []
    for value in _values(request):
        total *= value
        result.append(total)
    return _list_result(result)


def prefix_minima(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _values(request)
    result = [values[0]]
    for value in values[1:]:
        result.append(min(result[-1], value))
    return _list_result(result)


def prefix_maxima(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _values(request)
    result = [values[0]]
    for value in values[1:]:
        result.append(max(result[-1], value))
    return _list_result(result)


def prefix_gcds(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _values(request)
    result = [values[0]]
    for value in values[1:]:
        result.append(math.gcd(result[-1], value))
    return _list_result(result)


def prefix_lcms(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    values = _values(request)
    result = [values[0]]
    for value in values[1:]:
        result.append(math.lcm(result[-1], value))
    return _list_result(result)


def sorted_unique(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    return _list_result(sorted(set(_values(request))))


def sort_sequence(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    return _list_result(sorted(_values(request)))


def reverse_sequence(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    return _list_result(list(reversed(_values(request))))


def parities(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    return _list_result([value % 2 for value in _values(request)])


def signs(request: IntegerSequenceRequest) -> IntegerSequenceListResult:
    return _list_result([(value > 0) - (value < 0) for value in _values(request)])


def frequencies(request: IntegerSequenceRequest) -> IntegerSequenceFrequenciesResult:
    counts = Counter(_values(request))
    entries = tuple(
        FrequencyEntry(value=str(value), count=counts[value])
        for value in sorted(counts)
    )
    return IntegerSequenceFrequenciesResult(entries=entries)


def zero_indices(request: IntegerSequenceRequest) -> IntegerSequenceIndexListResult:
    return IntegerSequenceIndexListResult(
        indices=tuple(
            index for index, value in enumerate(_values(request)) if value == 0
        ),
    )


def decide_arithmetic(request: IntegerSequenceRequest) -> IntegerSequenceBooleanResult:
    values = _values(request)
    if len(values) < 2:
        return IntegerSequenceBooleanResult(holds=True)
    differences = {right - left for left, right in pairwise(values)}
    return IntegerSequenceBooleanResult(holds=len(differences) <= 1)


def decide_geometric(request: IntegerSequenceRequest) -> IntegerSequenceBooleanResult:
    values = _values(request)
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
    values = _values(request)
    return IntegerSequenceBooleanResult(
        holds=all(left <= right for left, right in pairwise(values))
    )


def decide_strictly_increasing(
    request: IntegerSequenceRequest,
) -> IntegerSequenceBooleanResult:
    values = _values(request)
    return IntegerSequenceBooleanResult(
        holds=all(left < right for left, right in pairwise(values))
    )
