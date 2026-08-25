"""Public declaration for the exact bounded friable-count operation."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory._friable_operations import compute_friable_count
from jacobian.math.number_theory._models import (
    FriableCountRequest,
    FriableCountResult,
)
from jacobian.math.number_theory._support import number_theory_operation

FRIABLE_COUNT_OPERATION = number_theory_operation(
    "number_theory.friable.count.compute",
    "Count friable integers exactly",
    (
        "Return the exact number Psi(x, y) of positive integers at most x whose "
        "prime factors are all at most the inclusive cutoff y. The result retains "
        "x and y and replays the count inside the admitted materialized or "
        "generated-search envelope."
    ),
    FriableCountRequest,
    FriableCountResult,
    compute_friable_count,
    "number-theory",
    "friable",
    "smooth-number",
    "counting",
    "exact",
    examples=(
        example(
            "five_friable_through_100",
            (
                "Count the positive 5-friable integers at most 100; x and y "
                "must be canonical nonnegative decimals and the selected exact "
                "counting regime must fit its work budget."
            ),
            {"x": "100", "y": "5"},
        ),
    ),
)

__all__ = ["FRIABLE_COUNT_OPERATION"]
