from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.number_theory.sequences.core.values import (
    MAX_INTEGER_SEQUENCE_ITEM_DIGITS,
    MAX_SEQUENCE_TOTAL_DIGITS,
    IntegerSequence,
)


@pytest.mark.parametrize("sign", ["", "-"])
def test_integer_sequence_accepts_items_at_exact_digit_bound(sign: str) -> None:
    value = parse_canonical_integer(sign + "1" * MAX_INTEGER_SEQUENCE_ITEM_DIGITS)

    request = IntegerSequence(values=(value,))

    assert request.values == (value,)


@pytest.mark.parametrize("sign", ["", "-"])
def test_integer_sequence_rejects_items_over_digit_bound(sign: str) -> None:
    value = parse_canonical_integer(sign + "1" * (MAX_INTEGER_SEQUENCE_ITEM_DIGITS + 1))

    with pytest.raises(ValidationError) as exc_info:
        IntegerSequence(values=(value,))

    assert exc_info.value.errors()[0]["type"] in {
        "sequences.item_too_large",
        "exact_integer.digit_bound",
    }


def test_integer_sequence_bounds_total_mathematical_representation() -> None:
    value = parse_canonical_integer("1" * MAX_INTEGER_SEQUENCE_ITEM_DIGITS)
    accepted_count = MAX_SEQUENCE_TOTAL_DIGITS // len(
        format_canonical_integer(abs(value))
    )

    assert IntegerSequence(values=(value,) * accepted_count)
    with pytest.raises(ValidationError) as exc_info:
        IntegerSequence(values=(value,) * (accepted_count + 1))

    assert exc_info.value.errors()[0]["type"] == "sequences.representation_too_large"
