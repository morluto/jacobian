"""Finite divisibility poset construction and declaration."""

from __future__ import annotations

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.sets._models import FiniteIntegerSet
from jacobian.math.combinatorics.posets.core._models import (
    MAX_ELEMENT_LABEL_LENGTH,
    MAX_POSET_ELEMENTS,
    FinitePoset,
    IncomparablePair,
    OrderedPair,
    _transitive_reduction,
    canonical_poset_ranks,
    finite_poset_digest,
)
from jacobian.math.number_theory._divisibility_poset_kernels import (
    construct_divisibility_poset,
)
from jacobian.math.number_theory._divisibility_poset_models import (
    MAX_DIVISIBILITY_SET_SIZE,
    DivisibilityPosetRequest,
)
from jacobian.math.number_theory._support import number_theory_operation


def _admit_values(values: FiniteIntegerSet) -> tuple[str, ...]:
    if not isinstance(values, FiniteIntegerSet):
        raise OperationDomainValidationError(
            location=("values",),
            code="divisibility_poset.values_type",
            message="values must be a canonical finite set of positive integers",
        )
    elements = values.elements
    if not 0 <= len(elements) <= min(MAX_DIVISIBILITY_SET_SIZE, MAX_POSET_ELEMENTS):
        raise OperationDomainValidationError(
            location=("values",),
            code="divisibility_poset.values_size",
            message=(
                "values must contain between 0 and "
                f"{min(MAX_DIVISIBILITY_SET_SIZE, MAX_POSET_ELEMENTS)} distinct integers"
            ),
        )
    if any(
        type(value) is not str or len(value) > MAX_ELEMENT_LABEL_LENGTH
        for value in elements
    ):
        raise OperationDomainValidationError(
            location=("values",),
            code="divisibility_poset.values_label_length",
            message=(
                "values must use positive canonical integers no longer than "
                f"{MAX_ELEMENT_LABEL_LENGTH} characters"
            ),
        )
    try:
        parsed = tuple(parse_canonical_integer(value) for value in elements)
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
    return elements


def divisibility_poset(values: FiniteIntegerSet) -> FinitePoset:
    """Return the canonical proper-divisibility poset of positive integers."""
    admitted_values = _admit_values(values)
    data = construct_divisibility_poset(admitted_values)
    elements = tuple(sorted(admitted_values))
    strict = set(data.strict_order_pairs)
    covers = _transitive_reduction(elements, strict)
    strict_pairs = tuple(
        OrderedPair(lower=lower, upper=upper) for lower, upper in sorted(strict)
    )
    cover_pairs = tuple(
        OrderedPair(lower=lower, upper=upper) for lower, upper in sorted(covers)
    )
    incomparable = tuple(
        IncomparablePair(left=left, right=right)
        for index, left in enumerate(elements)
        for right in elements[index + 1 :]
        if (left, right) not in strict and (right, left) not in strict
    )
    minimal = tuple(
        element
        for element in elements
        if not any(upper == element for _, upper in strict)
    )
    maximal = tuple(
        element
        for element in elements
        if not any(lower == element for lower, _ in strict)
    )
    ranks = canonical_poset_ranks(elements, covers)
    digest = finite_poset_digest(
        elements=elements,
        strict_order_pairs=strict_pairs,
        cover_relations=cover_pairs,
        incomparable_pairs=incomparable,
        minimal_elements=minimal,
        maximal_elements=maximal,
        graded=ranks is not None,
        ranks=ranks,
    )
    return FinitePoset.model_construct(
        elements=elements,
        strict_order_pairs=strict_pairs,
        cover_relations=cover_pairs,
        incomparable_pairs=incomparable,
        minimal_elements=minimal,
        maximal_elements=maximal,
        graded=ranks is not None,
        ranks=ranks,
        poset_digest=digest,
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
            {"values": {"elements": ["2", "3", "6"]}},
        ),
    ),
)


__all__ = [
    "DIVISIBILITY_POSET_OPERATION",
    "compute_divisibility_poset",
    "divisibility_poset",
]
