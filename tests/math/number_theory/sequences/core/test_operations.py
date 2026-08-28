from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.sequences.core.operations import (
    decide_arithmetic,
    decide_geometric,
    decide_nondecreasing,
    decide_strictly_increasing,
    first_differences,
    frequencies,
    parities,
    prefix_gcds,
    prefix_lcms,
    prefix_maxima,
    prefix_minima,
    prefix_products,
    prefix_sums,
    reverse_sequence,
    second_differences,
    sequence_distinct_count,
    sequence_gcd,
    sequence_lcm,
    sequence_maximum,
    sequence_mean,
    sequence_median,
    sequence_minimum,
    sequence_product,
    sequence_range,
    sequence_sum,
    signs,
    sort_sequence,
    sorted_unique,
    zero_indices,
)
from jacobian.math.number_theory.sequences.core.values import (
    MAX_INTEGER_SEQUENCE_ITEM_DIGITS,
    IntegerSequence,
)


def test_native_aggregate_and_statistic_values_are_exact() -> None:
    sequence = IntegerSequence(values=("3", "1", "2", "2"))

    assert sequence_sum(sequence).value == "8"
    assert sequence_product(sequence).value == "12"
    assert sequence_gcd(sequence).value == "1"
    assert sequence_lcm(sequence).value == "6"
    assert sequence_minimum(sequence).value == "1"
    assert sequence_maximum(sequence).value == "3"
    assert sequence_range(sequence).value == "2"
    assert sequence_distinct_count(sequence).value == "3"
    assert sequence_mean(sequence).value.as_integer_ratio() == (2, 1)
    assert sequence_median(sequence).value.as_integer_ratio() == (2, 1)


def test_native_transforms_search_and_predicates_preserve_order() -> None:
    sequence = IntegerSequence(values=("3", "0", "1", "3"))

    assert prefix_sums(sequence).values == ("3", "3", "4", "7")
    assert first_differences(sequence).values == ("-3", "1", "2")
    assert second_differences(sequence).values == ("4", "1")
    assert prefix_products(sequence).values == ("3", "0", "0", "0")
    assert prefix_minima(sequence).values == ("3", "0", "0", "0")
    assert prefix_maxima(sequence).values == ("3", "3", "3", "3")
    assert prefix_gcds(sequence).values == ("3", "3", "1", "1")
    assert prefix_lcms(sequence).values == ("3", "0", "0", "0")
    assert sorted_unique(sequence).values == ("0", "1", "3")
    assert sort_sequence(sequence).values == ("0", "1", "3", "3")
    assert reverse_sequence(sequence).values == ("3", "1", "0", "3")
    assert parities(sequence).values == ("1", "0", "1", "1")
    assert signs(sequence).values == ("1", "0", "1", "1")
    assert frequencies(sequence).entries[0].value == "0"
    assert zero_indices(sequence).indices == (1,)
    assert decide_arithmetic(IntegerSequence(values=("1", "3", "5"))).holds
    assert decide_geometric(IntegerSequence(values=("2", "4", "8"))).holds
    assert decide_nondecreasing(IntegerSequence(values=("1", "1", "3"))).holds
    assert decide_strictly_increasing(IntegerSequence(values=("1", "2", "3"))).holds


def test_native_admission_rejects_unrepresentable_product() -> None:
    sequence = IntegerSequence(values=("9" * 20_000, "9" * 20_000))

    with pytest.raises(OperationDomainValidationError) as error:
        sequence_product(sequence)

    assert error.value.errors()[0]["type"] == "sequences.result_digits_exceeded"


def test_zero_absorbs_later_prefix_product_and_lcm_widths() -> None:
    sequence = IntegerSequence(
        values=("0", "1" + "0" * (MAX_INTEGER_SEQUENCE_ITEM_DIGITS - 1))
    )

    assert prefix_products(sequence).values == ("0", "0")
    assert prefix_lcms(sequence).values == ("0", "0")
