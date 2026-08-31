"""Exact bounded r-full integer enumeration and declaration."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._r_full_enumerate_kernels import (
    enumerate_r_full as enumerate_r_full_kernel,
)
from jacobian.math.number_theory._r_full_enumerate_models import (
    MAX_R_FULL_CUTOFF,
    MIN_R_FULL_EXPONENT,
    RFullEnumerateRequest,
    RFullEnumerateResult,
    max_r_full_exponent,
    plan_r_full_family,
)
from jacobian.math.number_theory._support import number_theory_operation


def _enumerate_r_full_admitted(
    minimum_exponent: int, cutoff: int, *, enforce_transport: bool
) -> RFullEnumerateResult:
    canonical_cutoff = format_canonical_integer(cutoff)
    plan = plan_r_full_family(minimum_exponent, cutoff)
    if plan.exceeded:
        if plan.reason == "planning":
            raise OperationDomainValidationError(
                location=("cutoff",),
                code="r_full_enumerate_planning_work_exceeds_budget",
                message="r-full planning work exceeds the admitted budget",
            )
        if plan.reason == "bytes":
            raise OperationDomainValidationError(
                location=("cutoff",),
                code="r_full_enumerate_family_exceeds_transport_budget",
                message="r-full family exceeds the serialized-byte budget",
            )
        raise OperationDomainValidationError(
            location=("cutoff",),
            code="r_full_enumerate_family_exceeds_result_budget",
            message="r-full family exceeds the result-size budget",
        )
    family = enumerate_r_full_kernel(
        cutoff, minimum_exponent, planned_family=plan.family
    )
    return RFullEnumerateResult._from_kernel(minimum_exponent, canonical_cutoff, family)


def enumerate_r_full_numbers(
    request: RFullEnumerateRequest,
) -> RFullEnumerateResult:
    """Return the complete ordered family of r-full integers up to the cutoff."""
    return _enumerate_r_full_admitted(
        request.minimum_exponent,
        parse_canonical_integer(request.cutoff),
        enforce_transport=True,
    )


def enumerate_r_full(minimum_exponent: int, cutoff: int) -> tuple[int, ...]:
    """Enumerate r-full integers for native callers using integer arguments.

    Returns the complete ordered family as a tuple of Python integers, so
    callers receive directly composable mathematical values rather than a
    wire-format transport model.
    """
    if type(minimum_exponent) is not int or not (
        MIN_R_FULL_EXPONENT <= minimum_exponent
    ):
        raise OperationDomainValidationError(
            location=("minimum_exponent",),
            code="r_full_enumerate.exponent_bound",
            message=(
                f"minimum_exponent must be an integer at least {MIN_R_FULL_EXPONENT}"
            ),
        )
    if type(cutoff) is not int or not (0 < cutoff <= MAX_R_FULL_CUTOFF):
        raise OperationDomainValidationError(
            location=("cutoff",),
            code="r_full_enumerate.cutoff_bound",
            message="cutoff must be a positive integer within the admitted bound",
        )
    max_exp = max_r_full_exponent(cutoff)
    if minimum_exponent > max_exp:
        raise OperationDomainValidationError(
            location=("minimum_exponent",),
            code="r_full_enumerate.exponent_bound",
            message=(
                f"minimum_exponent {minimum_exponent} exceeds the maximum "
                f"meaningful exponent {max_exp} for cutoff {cutoff}"
            ),
        )
    plan = plan_r_full_family(minimum_exponent, cutoff)
    if plan.exceeded:
        if plan.reason == "planning":
            raise OperationDomainValidationError(
                location=("cutoff",),
                code="r_full_enumerate_planning_work_exceeds_budget",
                message="r-full planning work exceeds the admitted budget",
            )
        raise OperationDomainValidationError(
            location=("cutoff",),
            code="r_full_enumerate_family_exceeds_result_budget",
            message="r-full family exceeds the result-size budget",
        )
    return plan.family


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
            {"minimum_exponent": 3, "cutoff": "20"},
        ),
    ),
)


__all__ = ["R_FULL_ENUMERATE_OPERATION", "enumerate_r_full", "enumerate_r_full_numbers"]
