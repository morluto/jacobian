"""Exact bounded r-full integer enumeration and declaration."""

from __future__ import annotations

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._r_full_enumerate_kernels import enumerate_r_full
from jacobian.math.number_theory._r_full_enumerate_models import (
    MAX_R_FULL_FAMILY_SIZE,
    MAX_R_FULL_RESULT_BYTES,
    RFullEnumerateRequest,
    RFullEnumerateResult,
)
from jacobian.math.number_theory._support import number_theory_operation


def enumerate_r_full_numbers(
    request: RFullEnumerateRequest,
) -> RFullEnumerateResult:
    """Return the complete ordered family of r-full integers up to the cutoff."""
    cutoff = parse_canonical_integer(request.cutoff)
    estimate = 10 * (cutoff ** (1 / request.minimum_exponent))
    if estimate > MAX_R_FULL_FAMILY_SIZE:
        raise OperationDomainValidationError(
            location=("cutoff",),
            code="r_full_enumerate_family_exceeds_result_budget",
            message="r-full family exceeds the result-size budget",
        )
    if estimate * (len(request.cutoff) + 3) > MAX_R_FULL_RESULT_BYTES:
        raise OperationDomainValidationError(
            location=("cutoff",),
            code="r_full_enumerate_family_exceeds_transport_budget",
            message="r-full family exceeds the serialized-byte budget",
        )
    raw_family = enumerate_r_full(cutoff, request.minimum_exponent)
    return RFullEnumerateResult._from_kernel(
        request.minimum_exponent, request.cutoff, raw_family
    )


R_FULL_ENUMERATE_OPERATION = number_theory_operation(
    "integer.r_full.enumerate",
    "Enumerate bounded r-full integers",
    "Given a minimum exponent r and a positive upper bound, return every r-full "
    "integer in [1, cutoff] exactly once in increasing order. An integer is r-full "
    "when every prime factor occurs to exponent at least r. Uses multiplicative "
    "generation from prime powers to avoid scanning every integer.",
    RFullEnumerateRequest,
    RFullEnumerateResult,
    enumerate_r_full_numbers,
    "number-theory",
    "r-full",
    "powerful",
    "enumerate",
    "exact",
    examples=(
        example(
            "three_full_to_20",
            "Enumerate every 3-full (cubefull) integer up to 20; the cutoff "
            "must be a positive integer and minimum_exponent must be at least 2.",
            {"minimum_exponent": 3, "cutoff": 20},
        ),
    ),
)


__all__ = ["R_FULL_ENUMERATE_OPERATION", "enumerate_r_full_numbers"]
