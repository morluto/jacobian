from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.number_theory.sequences.core.values import (
    MAX_INTEGER_SEQUENCE_ITEM_DIGITS,
    IntegerSequence,
)


@pytest.mark.parametrize("sign", ["", "-"])
def test_integer_sequence_accepts_items_at_exact_digit_bound(sign: str) -> None:
    value = sign + "1" * MAX_INTEGER_SEQUENCE_ITEM_DIGITS

    request = IntegerSequence(values=(value,))

    assert request.values == (value,)


@pytest.mark.parametrize("sign", ["", "-"])
def test_integer_sequence_rejects_items_over_digit_bound(sign: str) -> None:
    value = sign + "1" * (MAX_INTEGER_SEQUENCE_ITEM_DIGITS + 1)

    with pytest.raises(ValidationError) as exc_info:
        IntegerSequence(values=(value,))

    assert exc_info.value.errors()[0]["type"] == "sequences.item_too_large"
