"""Counting-family model contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics._counting_models import (
    MAX_BINOMIAL_N,
    BinomialRequest,
    IntegerListRequest,
)


def test_binomial_request_retains_the_wider_exact_bound() -> None:
    schema = BinomialRequest.model_json_schema()

    assert schema["properties"]["n"]["maximum"] == MAX_BINOMIAL_N
    assert BinomialRequest(n=MAX_BINOMIAL_N, k=0).model_dump() == {
        "n": MAX_BINOMIAL_N,
        "k": 0,
    }


def test_integer_list_accepts_canonical_values_beyond_python_limit() -> None:
    value = "1" + ("0" * 5_000)

    request = IntegerListRequest(values=(value,))

    assert request.values == (value,)


def test_integer_list_retains_the_nonnegative_parts_contract() -> None:
    with pytest.raises(ValidationError) as exc_info:
        IntegerListRequest(values=("-1",))

    assert exc_info.value.errors()[0]["type"] == "combinatorics.invariant"
