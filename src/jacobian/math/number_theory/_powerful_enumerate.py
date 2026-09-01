"""Exact bounded powerful-number enumeration and declaration."""

from __future__ import annotations

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory._powerful_enumerate_kernels import (
    enumerate_powerful,
)
from jacobian.math.number_theory._powerful_enumerate_models import (
    PowerfulEnumerateRequest,
    PowerfulEnumerateResult,
)


def enumerate_powerful_numbers(
    request: PowerfulEnumerateRequest,
) -> PowerfulEnumerateResult:
    """Return the complete ordered family of powerful integers up to the cutoff."""

    raw_family = enumerate_powerful(request.cutoff)
    return PowerfulEnumerateResult._from_kernel(request.cutoff, raw_family)


POWERFUL_ENUMERATE_OPERATION = MathTool(
    operation_id="integer.powerful.enumerate",
    title="Enumerate bounded powerful integers",
    description="Given a positive upper bound, return every powerful integer in [1, cutoff] "
    "exactly once in increasing order, using the canonical square-cube "
    "representation n = a^2 * b^3 with b squarefree. The unique representation "
    "guarantees completeness and no duplicates.",
    request_type=PowerfulEnumerateRequest,
    result_type=PowerfulEnumerateResult,
    run=enumerate_powerful_numbers,
    tags=(
        "number-theory",
        "2-full",
        "powerful",
        "powerful-number",
        "enumerate",
        "exact",
    ),
    examples=(
        OperationExample(
            name="powerful_to_100",
            description="Enumerate every powerful integer up to 100; the cutoff must be a "
            "positive integer.",
            input={"cutoff": 100},
        ),
    ),
)
