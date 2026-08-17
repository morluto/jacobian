from __future__ import annotations

import sys
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.math.probability._models import (
    GraphConnectionProbabilityResult,
    GraphReliabilityEdgeProbability,
)


def _rational(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


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


@pytest.mark.parametrize(
    ("state_probabilities", "connection_probability"),
    (((-1, 2), 2), ((2, -1), -1)),
    ids=("negative-first", "over-one-first"),
)
def test_graph_reliability_rejects_out_of_range_state_probabilities_that_sum_to_one(
    state_probabilities: tuple[int, int],
    connection_probability: int,
) -> None:
    assert sum((Fraction(value) for value in state_probabilities), Fraction()) == 1

    with pytest.raises(
        ValidationError,
        match=r"graph reliability state probability must lie in \[0, 1\]",
    ):
        GraphConnectionProbabilityResult.model_validate(
            {
                "terminals": ["a", "b"],
                "connection_probability": _rational(connection_probability),
                "edge_count": 1,
                "visited_states": 2,
                "states": [
                    {
                        "state_index": 0,
                        "open_edges": [],
                        "terminals_connected": False,
                        "state_probability": _rational(state_probabilities[0]),
                    },
                    {
                        "state_index": 1,
                        "open_edges": [["a", "b"]],
                        "terminals_connected": True,
                        "state_probability": _rational(state_probabilities[1]),
                    },
                ],
            }
        )
