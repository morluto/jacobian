"""Divisibility edge profile with quotient and LPF declaration."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.canonical import CanonicalizationError, format_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._divisibility_edge_profile_kernels import (
    FactorizationIncompleteError,
    construct_divisibility_edge_profile,
)
from jacobian.math.number_theory._divisibility_edge_profile_models import (
    MAX_DIVISIBILITY_EDGE_VALUE_DIGITS,
    DivisibilityEdge,
    DivisibilityEdgeProfileRequest,
    DivisibilityEdgeProfileResult,
    _validate_divisibility_edge_resources,
    _validate_divisibility_edge_shape,
)
from jacobian.math.number_theory._support import number_theory_operation
from jacobian.math.number_theory.arithmetic.values import IntegerValue


def compute_divisibility_edge_profile(
    request: DivisibilityEdgeProfileRequest,
) -> DivisibilityEdgeProfileResult:
    """Return the complete directed divisibility edge table with quotient and LPF."""
    try:
        _validate_divisibility_edge_resources(request.values)
        return _build_divisibility_edge_profile(request.values)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("values",), code=exc.type, message=exc.message()
        ) from exc
    except FactorizationIncompleteError as exc:
        failure = exc.failure
        failure_kind = failure.kind if failure is not None else "UNKNOWN"
        raise RuntimeError(
            "divisibility edge factorization worker failed: " + failure_kind
        ) from exc


def _build_divisibility_edge_profile(
    values: tuple[str, ...],
) -> DivisibilityEdgeProfileResult:
    try:
        data = construct_divisibility_edge_profile(values)
    except FactorizationIncompleteError as exc:
        failure = exc.failure
        if failure is not None and failure.kind == "WORKER_CANCELLED":
            raise OperationExecutionCancelledError(
                "divisibility edge factorization was cancelled"
            ) from exc
        if failure is not None and failure.kind in {
            "WORKER_TIMEOUT",
            "REQUEST_DEADLINE_EXPIRED",
        }:
            raise OperationExecutionTimeoutError(
                "divisibility edge factorization exceeded its deadline"
            ) from exc
        raise OperationDomainValidationError(
            location=("values",),
            code="divisibility_edge.factorization_incomplete",
            message="bounded factorization did not establish every quotient",
        ) from exc
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
    try:
        canonical_values = tuple(_canonical_native_value(value) for value in values)
        _validate_divisibility_edge_shape(canonical_values)
        _validate_divisibility_edge_resources(canonical_values)
    except (CanonicalizationError, PydanticCustomError, TypeError, ValueError) as exc:
        if isinstance(exc, PydanticCustomError):
            code = exc.type
            message = exc.message()
        else:
            code = "divisibility_edge.invalid_native_value"
            message = "values must be canonical integers or IntegerValue instances"
        raise OperationDomainValidationError(
            location=("values",), code=code, message=message
        ) from exc
    return _build_divisibility_edge_profile(canonical_values)


def _canonical_native_value(value: str | int | IntegerValue) -> str:
    if isinstance(value, IntegerValue):
        return value.value
    if type(value) is str:
        return value
    if type(value) is int:
        if abs(value) >= 10**MAX_DIVISIBILITY_EDGE_VALUE_DIGITS:
            raise PydanticCustomError(
                "divisibility_edge.value_digits",
                "values exceed the admitted integer digit bound",
            )
        return format_canonical_integer(value)
    raise TypeError("native divisibility values must be strings or integers")


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
