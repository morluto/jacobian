"""Finite divisibility poset construction and declaration."""

from __future__ import annotations

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.posets.core._models import (
    FinitePoset,
    PresentationPair,
    ReflexivePairPolicy,
    RelationInterpretation,
)
from jacobian.math.combinatorics.posets.core.operations import materialize_finite_poset
from jacobian.math.number_theory._divisibility_poset_kernels import (
    construct_divisibility_poset,
)
from jacobian.math.number_theory._divisibility_poset_models import (
    MAX_DIVISIBILITY_SET_SIZE,
    DivisibilityPosetRequest,
)
from jacobian.math.number_theory._support import number_theory_operation


def _admit_values(values: tuple[str, ...]) -> None:
    if type(values) is not tuple:
        raise OperationDomainValidationError(
            location=("values",),
            code="divisibility_poset.values_type",
            message="values must be a tuple of canonical positive integers",
        )
    if not 1 <= len(values) <= MAX_DIVISIBILITY_SET_SIZE:
        raise OperationDomainValidationError(
            location=("values",),
            code="divisibility_poset.values_size",
            message=(
                "values must contain between 1 and "
                f"{MAX_DIVISIBILITY_SET_SIZE} distinct integers"
            ),
        )
    try:
        parsed = tuple(parse_canonical_integer(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("values",),
            code="divisibility_poset.values_domain",
            message="values must be canonical positive integers",
        ) from exc
    if any(value <= 0 for value in parsed):
        raise OperationDomainValidationError(
            location=("values",),
            code="divisibility_poset.values_domain",
            message="values must be canonical positive integers",
        )
    if len(set(values)) != len(values):
        raise OperationDomainValidationError(
            location=("values",),
            code="divisibility_poset.values_unique",
            message="values must be distinct",
        )


def divisibility_poset(values: tuple[str, ...]) -> FinitePoset:
    """Return the canonical proper-divisibility poset of positive integers."""
    _admit_values(values)
    data = construct_divisibility_poset(values)
    return materialize_finite_poset(
        tuple(sorted(values)),
        tuple(
            PresentationPair(lower=lower, upper=upper)
            for lower, upper in data.strict_order_pairs
        ),
        RelationInterpretation.COMPARABLE_PAIRS,
        ReflexivePairPolicy.FORBIDDEN,
    )


def compute_divisibility_poset(request: DivisibilityPosetRequest) -> FinitePoset:
    """Adapt the wire request to the native divisibility-poset operation."""
    return divisibility_poset(request.values)


DIVISIBILITY_POSET_OPERATION = number_theory_operation(
    "integer.divisibility_poset.compute",
    "Construct finite divisibility poset",
    "Given a finite set of positive integers, return the canonical "
    "proper-divisibility poset where a < b exactly when a divides b "
    "and a != b. The result is a source-labelled directed relation.",
    DivisibilityPosetRequest,
    FinitePoset,
    compute_divisibility_poset,
    "number-theory",
    "divisibility",
    "poset",
    "exact",
    examples=(
        example(
            "divisibility_236",
            "For {2,3,6}, the proper-divisibility poset has 2<6 and 3<6; "
            "values must be positive canonical decimal integers.",
            {"values": ["2", "3", "6"]},
        ),
    ),
)


__all__ = [
    "DIVISIBILITY_POSET_OPERATION",
    "compute_divisibility_poset",
    "divisibility_poset",
]
