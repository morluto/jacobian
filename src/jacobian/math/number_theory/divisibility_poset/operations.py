"""Construct a canonical finite poset under proper divisibility."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.sets._models import (
    FiniteIntegerSet,
)
from jacobian.math.combinatorics.posets.core._models import (
    PresentationPair,
    ReflexivePairPolicy,
    RelationInterpretation,
)
from jacobian.math.combinatorics.posets.core.operations import (
    materialize_finite_poset,
)
from jacobian.math.number_theory.divisibility_poset._models import (
    ElementSource,
    IntegerDivisibilityPosetResult,
    _divisibility_poset_admission_error,
)


def compute_divisibility_poset(
    source_set: FiniteIntegerSet,
) -> IntegerDivisibilityPosetResult:
    """Build the canonical proper-divisibility poset from a positive integer set.

    Each source integer becomes a poset element.  The strict order is proper
    divisibility: ``x < y`` iff ``x`` divides ``y`` and ``x != y``.
    Generated labels are stable short identifiers (``e0``, ``e1``, …) so that
    the ``ElementLabel`` character cap never constrains source-integer digit
    length.
    """
    failure = _divisibility_poset_admission_error(source_set)
    if failure is not None:
        code, message = failure
        raise OperationDomainValidationError(
            location=("source_set",),
            code=f"number_theory.divisibility_poset.{code}",
            message=message,
        )
    values: list[int] = [parse_canonical_integer(v) for v in source_set.elements]

    # Sort by integer value for deterministic label assignment.  The source set
    # is already distinct (validated by FiniteIntegerSet), but it is not
    # required to arrive sorted.
    indexed = sorted(range(len(values)), key=lambda i: values[i])

    # Stable generated labels: e0, e1, e2, …
    label_for: dict[int, str] = {}
    label_to_value: dict[str, int] = {}
    for new_index, original_index in enumerate(indexed):
        label = f"e{new_index}"
        label_for[original_index] = label
        label_to_value[label] = values[original_index]

    elements = tuple(f"e{i}" for i in range(len(values)))

    # Build strict divisibility pairs (x divides y, x != y).
    pairs: list[PresentationPair] = []
    for i in range(len(values)):
        for j in range(len(values)):
            if i == j:
                continue
            x = values[i]
            y = values[j]
            if x == 0:
                continue
            if y % x == 0 and x != y:
                pairs.append(PresentationPair(lower=label_for[i], upper=label_for[j]))

    # Canonicalize: materialize_finite_poset with comparable-pair input
    # expects elements to be sorted.  The pairs must reference all declared
    # elements.
    materialized = materialize_finite_poset(
        elements=tuple(sorted(elements)),
        relation=tuple(sorted(pairs, key=lambda p: (p.lower, p.upper))),
        interpretation=RelationInterpretation.COMPARABLE_PAIRS,
        reflexive_pairs=ReflexivePairPolicy.FORBIDDEN,
    )

    element_sources = tuple(
        ElementSource(
            label=label,
            source_integer=format_canonical_integer(label_to_value[label]),
        )
        for label in materialized.elements
    )

    return IntegerDivisibilityPosetResult(
        source_set=source_set,
        poset=materialized,
        element_sources=element_sources,
    )


__all__ = ["compute_divisibility_poset"]
