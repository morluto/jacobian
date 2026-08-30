"""Exact bounded powerful-number enumeration and declaration."""

from __future__ import annotations

from jacobian.catalog._examples import example
from jacobian.math.number_theory._powerful_enumerate_kernels import (
    enumerate_powerful,
)
from jacobian.math.number_theory._powerful_enumerate_models import (
    PowerfulEnumerateRequest,
    PowerfulEnumerateResult,
)
from jacobian.math.number_theory._support import number_theory_operation


def enumerate_powerful_numbers(
    request: PowerfulEnumerateRequest,
) -> PowerfulEnumerateResult:
    """Return the complete ordered family of powerful integers up to the cutoff."""

    raw_family = enumerate_powerful(request.cutoff)
    return PowerfulEnumerateResult._from_kernel(request.cutoff, raw_family)


POWERFUL_ENUMERATE_OPERATION = number_theory_operation(
    "integer.powerful.enumerate",
    "Enumerate bounded powerful integers",
    "Given a positive upper bound, return every powerful integer in [1, cutoff] "
    "exactly once in increasing order, using the canonical square-cube "
    "representation n = a^2 * b^3 with b squarefree. The unique representation "
    "guarantees completeness and no duplicates.",
    PowerfulEnumerateRequest,
    PowerfulEnumerateResult,
    enumerate_powerful_numbers,
    "number-theory",
    "2-full",
    "powerful",
    "powerful-number",
    "enumerate",
    "exact",
    examples=(
        example(
            "powerful_to_100",
            "Enumerate every powerful integer up to 100; the cutoff must be a "
            "positive integer.",
            {"cutoff": 100},
        ),
    ),
)
