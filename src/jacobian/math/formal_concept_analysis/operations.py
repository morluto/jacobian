"""Exact native kernels for formal concept analysis."""

from __future__ import annotations

from .values import FormalContext

__all__ = [
    "attribute_closure",
    "attribute_derivation",
    "concept_from_attributes",
    "concept_from_objects",
    "concept_lattice",
    "enumerate_concepts",
    "object_closure",
    "object_derivation",
]


def object_derivation(
    ctx: FormalContext, objects: frozenset[int]
) -> frozenset[int]:
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

    Under standard FCA semantics, the derivation of the empty attribute set
    is every object.
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


def concept_from_objects(ctx: FormalContext, objects: frozenset[int]) -> dict[str, frozenset[int]]:
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


def enumerate_concepts(ctx: FormalContext) -> list[dict[str, frozenset[int]]]:
    """Return every formal concept exactly once by brute-force enumeration of
    all closed attribute intents."""
    n = len(ctx.attributes)
    closed_intents: set[frozenset[int]] = set()
    for mask in range(2 ** n):
        candidate = frozenset(i for i in range(n) if mask & (1 << i))
        closure = object_derivation(ctx, attribute_derivation(ctx, candidate))
        closed_intents.add(frozenset(closure))
    concepts = []
    for intent in closed_intents:
        extent = attribute_derivation(ctx, intent)
        concepts.append({"extent": extent, "intent": intent})
    return concepts


def _next_closed_intent(
    ctx: FormalContext, current: frozenset[int]
) -> frozenset[int] | None:
    """Find the next closed intent in lectic order using brute-force enumeration."""
    n = len(ctx.attributes)
    current_set = set(current)
    for i in range(n - 1, -1, -1):
        if i in current_set:
            continue
        candidate = frozenset(a for a in current_set if a < i)
        candidate = candidate | {i}
        closure = object_derivation(ctx, attribute_derivation(ctx, candidate))
        if (closure == candidate or candidate.issubset(closure)) and frozenset(closure) > current:
                return frozenset(closure)
    return None


def concept_lattice(  # noqa: C901
    ctx: FormalContext,
) -> dict[str, object]:
    """Return the concept lattice: concepts, partial order by extent inclusion,
    cover relation, top and bottom concepts."""
    concepts = enumerate_concepts(ctx)
    n = len(concepts)
    order: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if i != j:
                ext_i = concepts[i]["extent"]
                ext_j = concepts[j]["extent"]
                if ext_i.issubset(ext_j):
                    order.append((i, j))
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
