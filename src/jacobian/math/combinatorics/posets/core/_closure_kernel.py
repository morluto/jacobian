"""Worker-safe kernels for poset closure, dual, and subposet operations."""

from __future__ import annotations

from jacobian.math.combinatorics.posets.core._closure_models import (
    DualPosetResult,
    InducedSubposetResult,
    LowerClosureResult,
    UpperClosureResult,
)
from jacobian.math.combinatorics.posets.core._models import (
    FinitePoset,
    IncomparablePair,
    OrderedPair,
    canonical_poset_ranks,
    finite_poset_digest,
)


def _build_comparable_map(
    elements: tuple[str, ...],
    strict_pairs: tuple[OrderedPair, ...],
) -> dict[str, set[str]]:
    """Return {element: {elements <= it}} for x <= y."""
    below: dict[str, set[str]] = {e: {e} for e in elements}
    pair_set = {(p.lower, p.upper) for p in strict_pairs}
    for e in elements:
        for p in elements:
            if (p, e) in pair_set:
                below[e].add(p)
    return below


def _build_above_map(
    elements: tuple[str, ...],
    strict_pairs: tuple[OrderedPair, ...],
) -> dict[str, set[str]]:
    """Return {element: {elements >= it}} for x >= y."""
    above: dict[str, set[str]] = {e: {e} for e in elements}
    pair_set = {(p.lower, p.upper) for p in strict_pairs}
    for e in elements:
        for p in elements:
            if (e, p) in pair_set:
                above[e].add(p)
    return above


def lower_closure(poset: FinitePoset, subset: tuple[str, ...]) -> LowerClosureResult:
    """Compute ↓S = {x : x <= s for some s in S}."""
    below = _build_comparable_map(poset.elements, poset.strict_order_pairs)
    closure: set[str] = set()
    for s in subset:
        closure |= below[s]
    return LowerClosureResult(
        poset_digest=poset.poset_digest,
        subset=subset,
        closure=tuple(sorted(closure)),
    )


def upper_closure(poset: FinitePoset, subset: tuple[str, ...]) -> UpperClosureResult:
    """Compute ↑S = {x : s <= x for some s in S}."""
    above = _build_above_map(poset.elements, poset.strict_order_pairs)
    closure: set[str] = set()
    for s in subset:
        closure |= above[s]
    return UpperClosureResult(
        poset_digest=poset.poset_digest,
        subset=subset,
        closure=tuple(sorted(closure)),
    )


def dual_poset(poset: FinitePoset) -> DualPosetResult:
    """Compute the dual poset (order reversed)."""
    elements = tuple(sorted(poset.elements))
    # Swap lower/upper in every strict pair
    dual_strict = tuple(
        OrderedPair(lower=p.upper, upper=p.lower)
        for p in sorted(poset.strict_order_pairs, key=lambda p: (p.upper, p.lower))
    )
    dual_covers = tuple(
        OrderedPair(lower=p.upper, upper=p.lower)
        for p in sorted(poset.cover_relations, key=lambda p: (p.upper, p.lower))
    )
    # Compute incomparable pairs (same as original)
    strict_set = {(p.lower, p.upper) for p in dual_strict}
    incomparable = tuple(
        IncomparablePair(left=left, right=right)
        for i, left in enumerate(elements)
        for right in elements[i + 1 :]
        if (left, right) not in strict_set and (right, left) not in strict_set
    )
    # Recompute minimal/maximal (swap roles)
    dual_minimal = poset.maximal_elements
    dual_maximal = poset.minimal_elements
    # Compute ranks
    dual_ranks = canonical_poset_ranks(
        elements, {(p.lower, p.upper) for p in dual_covers}
    )
    # Compute digest
    dual_digest = finite_poset_digest(
        elements=elements,
        strict_order_pairs=dual_strict,
        cover_relations=dual_covers,
        incomparable_pairs=incomparable,
        minimal_elements=dual_minimal,
        maximal_elements=dual_maximal,
        graded=poset.graded,
        ranks=tuple(dual_ranks)
        if (dual_ranks is not None and poset.ranks is not None)
        else None,
    )
    dual = FinitePoset(
        elements=elements,
        strict_order_pairs=dual_strict,
        cover_relations=dual_covers,
        incomparable_pairs=incomparable,
        minimal_elements=dual_minimal,
        maximal_elements=dual_maximal,
        graded=poset.graded,
        ranks=tuple(dual_ranks)
        if (dual_ranks is not None and poset.ranks is not None)
        else None,
        poset_digest=dual_digest,
    )
    return DualPosetResult(
        poset=dual,
        transport_map=elements,
    )


def induced_subposet(
    poset: FinitePoset, subset: tuple[str, ...]
) -> InducedSubposetResult:
    """Compute the subposet induced by a subset of elements."""
    subset_set = set(subset)
    elements = tuple(sorted(subset_set))

    # Filter strict pairs to only those within the subset, then compute
    # the full transitive closure so that non-convex subsets still have
    # correct order relations (e.g. chain a<b<c restricted to {a,c} gives a<c).
    filtered_pairs = {
        (p.lower, p.upper)
        for p in poset.strict_order_pairs
        if p.lower in subset_set and p.upper in subset_set
    }
    from jacobian.math.combinatorics.posets.core._models import (
        _strict_closure,
        _transitive_reduction,
    )

    closure = _strict_closure(elements, filtered_pairs)
    strict_pairs = tuple(
        OrderedPair(lower=lower, upper=upper) for lower, upper in sorted(closure)
    )
    restricted_covers_set = _transitive_reduction(elements, closure)
    covers = tuple(
        OrderedPair(lower=lower, upper=upper)
        for lower, upper in sorted(restricted_covers_set)
    )
    # Compute incomparable pairs
    strict_set = {(p.lower, p.upper) for p in strict_pairs}
    incomparable = tuple(
        IncomparablePair(left=left, right=right)
        for i, left in enumerate(elements)
        for right in elements[i + 1 :]
        if (left, right) not in strict_set and (right, left) not in strict_set
    )
    # Compute minimal/maximal
    # minimal: no element is below it (no (x, e) in strict_pairs)
    # maximal: no element is above it (no (e, x) in strict_pairs)
    all_above = {p.upper for p in strict_pairs}  # elements that have something below
    all_below = {p.lower for p in strict_pairs}  # elements that have something above
    minimal = tuple(e for e in elements if e not in all_above)
    maximal = tuple(e for e in elements if e not in all_below)
    # Compute ranks
    ranks = canonical_poset_ranks(elements, {(p.lower, p.upper) for p in covers})
    ranked = ranks is not None
    # Compute digest
    digest = finite_poset_digest(
        elements=elements,
        strict_order_pairs=strict_pairs,
        cover_relations=covers,
        incomparable_pairs=incomparable,
        minimal_elements=minimal,
        maximal_elements=maximal,
        graded=ranked,
        ranks=tuple(ranks) if (ranks is not None and ranked) else None,
    )
    subposet = FinitePoset(
        elements=elements,
        strict_order_pairs=strict_pairs,
        cover_relations=covers,
        incomparable_pairs=incomparable,
        minimal_elements=minimal,
        maximal_elements=maximal,
        graded=ranked,
        ranks=tuple(ranks) if (ranks is not None and ranked) else None,
        poset_digest=digest,
    )
    return InducedSubposetResult(
        subposet=subposet,
        old_to_new=tuple(sorted(subset_set)),
    )
