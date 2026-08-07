from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from jacobian.contracts.probability import GraphReliabilityEdgeProbability


@pytest.mark.parametrize(
    ("num", "den"),
    (
        ("1" * (sys.int_info.default_max_str_digits + 1), "1"),
        ("1", "1" * (sys.int_info.default_max_str_digits + 1)),
    ),
    ids=("numerator", "denominator"),
)
def test_large_probability_reports_the_contract_digit_bound(
    num: str,
    den: str,
) -> None:
    previous_limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(sys.int_info.default_max_str_digits)
    try:
        with pytest.raises(
            ValidationError,
            match="graph reliability edge probability exceeds the 128-digit bound",
        ):
            GraphReliabilityEdgeProbability.model_validate(
                {
                    "edge": ["a", "b"],
                    "open_probability": {
                        "num": num,
                        "den": den,
                    },
                }
            )
    finally:
        sys.set_int_max_str_digits(previous_limit)
