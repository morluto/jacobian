from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.sequences import (
    MAX_INTEGER_SEQUENCE_ITEM_DIGITS,
    IntegerSequenceRequest,
)


@pytest.mark.parametrize("sign", ["", "-"])
def test_integer_sequence_accepts_items_at_exact_digit_bound(sign: str) -> None:
    value = sign + "1" * MAX_INTEGER_SEQUENCE_ITEM_DIGITS

    request = IntegerSequenceRequest(values=(value,))

    assert request.values == (value,)


@pytest.mark.parametrize("sign", ["", "-"])
def test_integer_sequence_rejects_items_over_digit_bound(sign: str) -> None:
    value = sign + "1" * (MAX_INTEGER_SEQUENCE_ITEM_DIGITS + 1)

    with pytest.raises(
        ValidationError, match=r"sequence item exceeds the .*digit bound"
    ):
        IntegerSequenceRequest(values=(value,))
