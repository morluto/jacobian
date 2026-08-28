"""Pure finite-context kernels shared by admission and operation adapters."""

from __future__ import annotations

from jacobian.math.combinatorics.posets.formal_concepts.values import FormalContext

MAX_CONCEPTS = 10_000


def object_derivation(ctx: FormalContext, objects: frozenset[int]) -> frozenset[int]:
    if not objects:
        return frozenset(range(len(ctx.attributes)))
    attributes = set(range(len(ctx.attributes)))
    for object_index in objects:
        if not 0 <= object_index < len(ctx.objects):
            raise ValueError("object index out of range")
        attributes &= {
            attribute for obj, attribute in ctx.incidence if obj == object_index
        }
    return frozenset(attributes)


def attribute_derivation(
    ctx: FormalContext, attributes: frozenset[int]
) -> frozenset[int]:
    if not attributes:
        return frozenset(range(len(ctx.objects)))
    objects = set(range(len(ctx.objects)))
    for attribute_index in attributes:
        if not 0 <= attribute_index < len(ctx.attributes):
            raise ValueError("attribute index out of range")
        objects &= {
            obj for obj, attribute in ctx.incidence if attribute == attribute_index
        }
    return frozenset(objects)


def attribute_closure(ctx: FormalContext, attributes: frozenset[int]) -> frozenset[int]:
    return object_derivation(ctx, attribute_derivation(ctx, attributes))


def _next_closure(
    ctx: FormalContext, current: frozenset[int], attribute_count: int
) -> frozenset[int] | None:
    current_set = set(current)
    for index in range(attribute_count - 1, -1, -1):
        if index in current_set:
            current_set.discard(index)
            continue
        candidate = {attribute for attribute in current_set if attribute < index}
        candidate.add(index)
        closure = attribute_closure(ctx, frozenset(candidate))
        closure_set = set(closure)
        if index in closure_set and {
            attribute for attribute in closure_set if attribute < index
        } == {attribute for attribute in current_set if attribute < index}:
            return frozenset(closure_set)
    return None


def enumerate_concept_pairs(
    ctx: FormalContext,
    *,
    limit: int = MAX_CONCEPTS,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Enumerate canonical ``(extent, intent)`` pairs, stopping at ``limit``."""

    concepts: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    current: frozenset[int] | None = attribute_closure(ctx, frozenset())
    while current is not None:
        extent = attribute_derivation(ctx, current)
        concepts.append((tuple(sorted(extent)), tuple(sorted(current))))
        if len(concepts) > limit:
            raise ValueError(
                f"concept count exceeds maximum of {limit}; narrow the context "
                "or reduce the number of attributes"
            )
        current = _next_closure(ctx, current, len(ctx.attributes))
    return tuple(concepts)


def concept_family_size_capped(ctx: FormalContext, limit: int) -> int:
    current: frozenset[int] | None = attribute_closure(ctx, frozenset())
    count = 0
    while current is not None:
        count += 1
        if count > limit:
            return count
        current = _next_closure(ctx, current, len(ctx.attributes))
    return count


def concept_lattice_from_pairs(
    concepts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
) -> dict[str, object]:
    extents = [frozenset(extent) for extent, _ in concepts]
    order = tuple(
        (left, right)
        for left, extent in enumerate(extents)
        for right, other in enumerate(extents)
        if left != right and extent.issubset(other)
    )
    order_set = set(order)
    covers = tuple(
        (left, right)
        for left, right in order
        if not any(
            middle != left
            and middle != right
            and (left, middle) in order_set
            and (middle, right) in order_set
            for middle in range(len(concepts))
        )
    )
    if not concepts:
        return {"concepts": (), "order": (), "covers": (), "top": None, "bottom": None}
    bottom = min(range(len(concepts)), key=lambda index: len(extents[index]))
    top = max(range(len(concepts)), key=lambda index: len(extents[index]))
    return {
        "concepts": concepts,
        "order": order,
        "covers": covers,
        "top": top,
        "bottom": bottom,
    }
