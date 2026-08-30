"""Divisibility edge profile with quotient and LPF declaration."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian.canonical import format_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._divisibility_edge_profile_kernels import (
    construct_divisibility_edge_profile,
)
from jacobian.math.number_theory._divisibility_edge_profile_models import (
    DivisibilityEdge,
    DivisibilityEdgeProfileRequest,
    DivisibilityEdgeProfileResult,
    _validate_divisibility_edge_values,
)
from jacobian.math.number_theory._support import number_theory_operation
from jacobian.math.number_theory.arithmetic.values import IntegerValue


def compute_divisibility_edge_profile(
    request: DivisibilityEdgeProfileRequest,
) -> DivisibilityEdgeProfileResult:
    """Return the complete directed divisibility edge table with quotient and LPF."""
    return _build_divisibility_edge_profile(request.values)


def _build_divisibility_edge_profile(
    values: tuple[str, ...],
) -> DivisibilityEdgeProfileResult:
    data = construct_divisibility_edge_profile(values)
    edges = tuple(
        DivisibilityEdge(
            source=d.source,
            target=d.target,
            quotient=format_canonical_integer(d.quotient),
            least_prime_factor=format_canonical_integer(d.least_prime_factor),
        )
        for d in data
    )
    return DivisibilityEdgeProfileResult(values=values, edges=edges)


def divisibility_edge_profile(
    values: tuple[str | int | IntegerValue, ...],
) -> DivisibilityEdgeProfileResult:
    """Return a divisibility edge profile from native canonical values."""
    canonical_values = tuple(
        value.value
        if isinstance(value, IntegerValue)
        else value
        if isinstance(value, str)
        else format_canonical_integer(value)
        for value in values
    )
    try:
        _validate_divisibility_edge_values(canonical_values)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("values",), code=exc.type, message=exc.message()
        ) from exc
    return _build_divisibility_edge_profile(canonical_values)


DIVISIBILITY_EDGE_PROFILE_OPERATION = number_theory_operation(
    "integer.divisibility_edge_profile.compute",
    "Profile quotient and least-prime-factor data on divisibility edges",
    "Given an ordered finite source set of positive integers, return the "
    "complete directed proper-divisibility edge table. Each edge a -> b "
    "carries the quotient b/a and its least prime factor.",
    DivisibilityEdgeProfileRequest,
    DivisibilityEdgeProfileResult,
    compute_divisibility_edge_profile,
    "number-theory",
    "divisibility",
    "primitive-set",
    "least-prime-factor",
    "exact",
    examples=(
        example(
            "divisibility_edges_24612",
            "For (2,4,6,12), profile all proper-divisibility edges with "
            "quotient and least-prime-factor data; values must be positive "
            "canonical decimal integers.",
            {"values": ["2", "4", "6", "12"]},
        ),
    ),
)


__all__ = [
    "DIVISIBILITY_EDGE_PROFILE_OPERATION",
    "compute_divisibility_edge_profile",
    "divisibility_edge_profile",
]
