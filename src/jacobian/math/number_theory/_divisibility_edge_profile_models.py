"""Typed contracts for divisibility edge profiles with quotient and LPF data."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.combinatorics.finite_structures.sets._models import FiniteIntegerSet

MAX_DIVISIBILITY_EDGE_SET_SIZE = 500
# LPF extraction uses the isolated direct-factorization worker for derived
# quotients; source values may use the canonical integer envelope.
MAX_DIVISIBILITY_EDGE_VALUE_DIGITS = 256
MAX_DIVISIBILITY_EDGE_QUOTIENT_DIGITS = 20
MAX_DIVISIBILITY_EDGE_WORK = 1_000_000
MAX_DIVISIBILITY_EDGE_PAIR_SCAN_WORK = 2_000_000
_FACTORIZATION_STARTUP_WORK = 10_000
MAX_DIVISIBILITY_EDGE_RESULT_BYTES = 10 * 1024 * 1024


class DivisibilityEdgeProfileRequest(StrictModel):
    """Profile quotient and least-prime-factor data on finite divisibility edges."""

    values: FiniteIntegerSet = Field(
        description=(
            "Ordered source set of positive canonical decimal integers. "
            f"Each value has at most {MAX_DIVISIBILITY_EDGE_VALUE_DIGITS} digits; "
            f"each derived quotient has at most {MAX_DIVISIBILITY_EDGE_QUOTIENT_DIGITS} digits. "
            "The result profiles every proper-divisibility edge a -> b "
            "(a divides b, a != b) with the quotient b/a and its least "
            "prime factor. "
            f"At most {MAX_DIVISIBILITY_EDGE_SET_SIZE} values are accepted."
        ),
        examples=[{"elements": ["2", "4", "6", "12"]}],
    )

    @model_validator(mode="after")
    def require_admitted_values(self) -> Self:
        _validate_divisibility_edge_shape(self.values)
        return self


def _extract_elements(values: object) -> tuple[str, ...]:
    """Extract the element tuple from a FiniteIntegerSet or raw tuple."""
    if isinstance(values, FiniteIntegerSet):
        return values.elements
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        return tuple(values)
    raise TypeError("values must be a FiniteIntegerSet or tuple of canonical integers")


def _validate_divisibility_edge_values(values: tuple[str, ...]) -> None:
    _validate_divisibility_edge_shape(values)
    _validate_divisibility_edge_resources(values)


def _validate_divisibility_edge_shape(values: object) -> tuple[str, ...]:
    elements = _extract_elements(values)
    if not elements:
        return ()
    if len(elements) > MAX_DIVISIBILITY_EDGE_SET_SIZE:
        raise PydanticCustomError(
            "divisibility_edge.values_size",
            f"values must contain at most {MAX_DIVISIBILITY_EDGE_SET_SIZE} integers",
        )
    if any(len(value) > MAX_DIVISIBILITY_EDGE_VALUE_DIGITS for value in elements):
        raise PydanticCustomError(
            "divisibility_edge.value_digits",
            "values exceed the admitted integer digit bound",
        )
    parsed = tuple(parse_canonical_integer(value) for value in elements)
    if any(value <= 0 for value in parsed):
        raise PydanticCustomError(
            "divisibility_edge.positive_values",
            "values must be positive canonical integers",
        )
    if len(set(elements)) != len(elements):
        raise PydanticCustomError(
            "divisibility_edge.values_unique", "values must be distinct"
        )
    return elements


def _validate_divisibility_edge_resources(
    values: object,
) -> tuple[tuple[str, ...], tuple[tuple[int, int, int], ...]]:
    """Return (canonicalized elements, edge plan) for the admitted values.

    The elements are canonicalized (sorted by integer value) so the same
    mathematical set produces the same edge plan and retained result
    regardless of presentation order.
    """
    elements = _extract_elements(values)
    if not elements:
        return (), ()
    # Canonicalize element order so the same mathematical set produces the
    # same edge plan and retained result regardless of presentation.
    elements = tuple(sorted(elements, key=int))
    parsed = tuple(parse_canonical_integer(value) for value in elements)
    digits = [len(v) for v in elements]
    pair_scan_work = sum(
        digits[i] * digits[j]
        for i in range(len(elements))
        for j in range(len(elements))
        if i != j
    )
    if pair_scan_work > MAX_DIVISIBILITY_EDGE_PAIR_SCAN_WORK:
        raise PydanticCustomError(
            "divisibility_edge.pair_scan_work",
            "divisibility pair scan exceeds the admitted work budget",
        )
    max_digits = max(digits)
    if len(elements) * (max_digits + 32) > MAX_DIVISIBILITY_EDGE_RESULT_BYTES:
        raise PydanticCustomError(
            "divisibility_edge.result_bytes",
            "divisibility edge profile exceeds the serialized-byte budget",
        )
    edge_plan: list[tuple[int, int, int]] = []
    distinct_quotients: set[int] = set()
    for left_index, left in enumerate(parsed):
        for right_index, right in enumerate(parsed):
            if left_index == right_index or right % left:
                continue
            quotient = right // left
            if quotient <= 1:
                continue
            if len(str(quotient)) > MAX_DIVISIBILITY_EDGE_QUOTIENT_DIGITS:
                raise PydanticCustomError(
                    "divisibility_edge.quotient_digits",
                    "derived divisibility quotients exceed the factorization worker limit",
                )
            edge_plan.append((left_index, right_index, quotient))
            distinct_quotients.add(quotient)
    factorization_work = sum(
        _FACTORIZATION_STARTUP_WORK + len(str(quotient)) ** 2
        for quotient in distinct_quotients
    )
    if factorization_work > MAX_DIVISIBILITY_EDGE_WORK:
        raise PydanticCustomError(
            "divisibility_edge.factorization_work",
            "divisibility factorization exceeds the admitted work budget",
        )
    if len(edge_plan) * (2 * max_digits + 96) > MAX_DIVISIBILITY_EDGE_RESULT_BYTES:
        raise PydanticCustomError(
            "divisibility_edge.result_bytes",
            "divisibility edge profile exceeds the serialized-byte budget",
        )
    return elements, tuple(edge_plan)


class DivisibilityEdge(StrictModel):
    """One proper-divisibility edge with quotient and least-prime-factor data."""

    source: CanonicalInteger
    target: CanonicalInteger
    quotient: CanonicalInteger
    least_prime_factor: CanonicalInteger

    @model_validator(mode="after")
    def require_positive_factors(self) -> Self:
        if parse_canonical_integer(self.quotient) <= 1:
            raise PydanticCustomError(
                "divisibility_edge.quotient_positive",
                "quotient must be greater than one",
            )
        if parse_canonical_integer(self.least_prime_factor) <= 1:
            raise PydanticCustomError(
                "divisibility_edge.least_prime_factor_positive",
                "least prime factor must be greater than one",
            )
        return self


class DivisibilityEdgeProfileResult(StrictModel):
    """The complete directed divisibility edge table."""

    values: FiniteIntegerSet = Field(
        description="The finite integer set that was profiled."
    )
    edges: tuple[DivisibilityEdge, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical_edges(self) -> Self:
        _validate_divisibility_edge_shape(self.values)
        values = set(self.values.elements)
        seen: set[tuple[str, str]] = set()
        for edge in self.edges:
            if edge.source == edge.target:
                raise PydanticCustomError(
                    "divisibility_edge.no_reflexive",
                    "divisibility edges must not be reflexive",
                )
            if edge.source not in values or edge.target not in values:
                raise PydanticCustomError(
                    "divisibility_edge.endpoints_declared",
                    "divisibility edge endpoints must occur in values",
                )
            key = (edge.source, edge.target)
            if key in seen:
                raise PydanticCustomError(
                    "divisibility_edge.edges_unique",
                    "divisibility edges must be unique",
                )
            seen.add(key)
        return self


__all__ = [
    "MAX_DIVISIBILITY_EDGE_QUOTIENT_DIGITS",
    "MAX_DIVISIBILITY_EDGE_SET_SIZE",
    "MAX_DIVISIBILITY_EDGE_VALUE_DIGITS",
    "DivisibilityEdge",
    "DivisibilityEdgeProfileRequest",
    "DivisibilityEdgeProfileResult",
    "_validate_divisibility_edge_resources",
    "_validate_divisibility_edge_values",
]
