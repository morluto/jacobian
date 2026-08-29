"""Product representation profile kernel."""

from __future__ import annotations

from jacobian.math.combinatorics.additive.product_representation._models import (
    ProductRepresentationResult,
    RepresentationEntry,
)
from jacobian.math.combinatorics.finite_structures.sets._models import FiniteIntegerSet

__all__ = ["compute_product_representation_profile"]


def compute_product_representation_profile(
    left: FiniteIntegerSet,
    right: FiniteIntegerSet,
) -> ProductRepresentationResult:
    """Return the complete exact product representation profile.

    For every x, the multiplicity r(x) = |{(a,b) in A x B : a*b = x}|.
    """
    counts: dict[int, int] = {}
    for a in left.elements:
        for b in right.elements:
            product = int(a) * int(b)
            counts[product] = counts.get(product, 0) + 1

    entries = tuple(
        RepresentationEntry(product=p, multiplicity=m)
        for p, m in sorted(counts.items())
    )

    return ProductRepresentationResult(
        left=left,
        right=right,
        entries=entries,
        support_cardinality=len(counts),
    )
