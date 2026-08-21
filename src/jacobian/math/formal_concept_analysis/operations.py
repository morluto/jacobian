"""Exact native kernels for formal concept analysis."""

from __future__ import annotations

from .values import FormalContext

# This bound is also enforced in the request models (_models.py).
MAX_CONCEPTS = 10000

__all__ = [
    "MAX_CONCEPTS",
    "attribute_closure",
    "attribute_derivation",
    "concept_from_attributes",
    "concept_from_objects",
    "concept_lattice",
    "enumerate_concepts",
    "object_closure",
    "object_derivation",
]


def object_derivation(ctx: FormalContext, objects: frozenset[int]) -> frozenset[int]:
    """Return A' = {m in M : every g in A has attribute m}.

    Under standard FCA semantics, the derivation of the empty object set is
    every attribute.
    """
    if not objects:
        return frozenset(range(len(ctx.attributes)))
    all_attrs: set[int] = set(range(len(ctx.attributes)))
    for oi in objects:
        if not 0 <= oi < len(ctx.objects):
            raise ValueError("object index out of range")
        attrs = {ai for o, ai in ctx.incidence if o == oi}
        all_attrs &= attrs
    return frozenset(all_attrs)


def attribute_derivation(
    ctx: FormalContext, attributes: frozenset[int]
) -> frozenset[int]:
    """Return B' = {g in G : every m in B is possessed by g}.

    Under standard FCA semantics, the derivation of the empty attribute set is
    every object.
    """
    if not attributes:
        return frozenset(range(len(ctx.objects)))
    all_objs: set[int] = set(range(len(ctx.objects)))
    for ai in attributes:
        if not 0 <= ai < len(ctx.attributes):
            raise ValueError("attribute index out of range")
        objs = {o for o, a in ctx.incidence if a == ai}
        all_objs &= objs
    return frozenset(all_objs)


def object_closure(ctx: FormalContext, objects: frozenset[int]) -> frozenset[int]:
    """Return A'' = (A')'."""
    return attribute_derivation(ctx, object_derivation(ctx, objects))


def attribute_closure(ctx: FormalContext, attributes: frozenset[int]) -> frozenset[int]:
    """Return B'' = (B')'."""
    return object_derivation(ctx, attribute_derivation(ctx, attributes))


def concept_from_objects(
    ctx: FormalContext, objects: frozenset[int]
) -> dict[str, frozenset[int]]:
    """Return the unique concept (A'', A')."""
    intent = object_derivation(ctx, objects)
    extent = attribute_derivation(ctx, intent)
    return {"extent": extent, "intent": intent}


def concept_from_attributes(
    ctx: FormalContext, attributes: frozenset[int]
) -> dict[str, frozenset[int]]:
    """Return the unique concept (B', B'')."""
    extent = attribute_derivation(ctx, attributes)
    intent = object_derivation(ctx, extent)
    return {"extent": extent, "intent": intent}


def _next_closure(
    ctx: FormalContext, current: frozenset[int], n: int
) -> frozenset[int] | None:
    """Find the next closed attribute set in lectic order after *current*.

    Implements Ganter's NextClosure algorithm.  The lectic order compares
    sets by scanning from the largest element downward: A < B iff the
    largest element where A and B differ belongs to B.
    """
    current_set = set(current)
    for i in range(n - 1, -1, -1):
        if i in current_set:
            current_set.discard(i)
            continue
        # Candidate = (current intersect {0,...,i-1}) union {i}
        candidate = {a for a in current_set if a < i}
        candidate.add(i)
        # closure = candidate'' (closure under the closure operator)
        closure = object_derivation(
            ctx, attribute_derivation(ctx, frozenset(candidate))
        )
        closure_set = set(closure)
        # Check lectic condition: closure agrees with current below i,
        # and i is in the closure (candidate is "licit-closed" up to i).
        # The standard condition is:
        #   closure intersect {0,...,i-1} == current intersect {0,...,i-1}  AND  i in closure
        if i not in closure_set:
            continue
        if {a for a in closure_set if a < i} != {a for a in current_set if a < i}:
            continue
        # closure is the next closed set in lectic order
        return frozenset(closure_set)
    return None


def enumerate_concepts(ctx: FormalContext) -> list[dict[str, frozenset[int]]]:
    """Return every formal concept exactly once using Ganter's NextClosure
    algorithm over the declared attribute order.

    The algorithm enumerates closed attribute intents in lectic order.
    Each step requires O(n) derivation operations, so the total cost is
    proportional to the number of concepts times n, not to 2^n.
    """
    n = len(ctx.attributes)
    concepts: list[dict[str, frozenset[int]]] = []

    # The empty set is always closed (it is the intent of the top concept).
    current: frozenset[int] | None = frozenset()
    while current is not None:
        intent = current
        extent = attribute_derivation(ctx, intent)
        concepts.append({"extent": extent, "intent": intent})
        if len(concepts) > MAX_CONCEPTS:
            raise ValueError(
                f"concept count exceeds maximum of {MAX_CONCEPTS}; "
                "narrow the context or reduce the number of attributes"
            )
        current = _next_closure(ctx, current, n)

    return concepts


def _inclusion_order(
    concepts: list[dict[str, frozenset[int]]],
) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    n = len(concepts)
    for i in range(n):
        ext_i = concepts[i]["extent"]
        for j in range(n):
            if i != j and ext_i.issubset(concepts[j]["extent"]):
                order.append((i, j))
    return order


def _cover_relation(order: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    order_set = set(order)
    covers: list[tuple[int, int]] = []
    for i, j in order:
        is_cover = True
        for k in range(n):
            if k != i and k != j and (i, k) in order_set and (k, j) in order_set:
                is_cover = False
                break
        if is_cover:
            covers.append((i, j))
    return covers


def concept_lattice(
    ctx: FormalContext,
) -> dict[str, object]:
    """Return the concept lattice: concepts, partial order by extent inclusion,
    cover relation, top and bottom concepts."""
    concepts = enumerate_concepts(ctx)
    n = len(concepts)
    order = _inclusion_order(concepts)
    covers = _cover_relation(order, n)
    if n == 0:
        return {"concepts": (), "order": (), "covers": (), "top": None, "bottom": None}
    bottom = 0
    top = 0
    for i in range(n):
        if concepts[i]["extent"] < concepts[bottom]["extent"]:
            bottom = i
        if concepts[i]["extent"] > concepts[top]["extent"]:
            top = i
    return {
        "concepts": tuple(concepts),
        "order": tuple(order),
        "covers": tuple(covers),
        "top": top,
        "bottom": bottom,
    }
