"""Pure finite-context kernels shared by admission and operation adapters."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.math.combinatorics.posets.formal_concepts.values import FormalContext

MAX_CONCEPTS = 10_000


@dataclass(frozen=True, slots=True)
class _IndexedDerivations:
    object_attributes: tuple[frozenset[int], ...]
    attribute_objects: tuple[frozenset[int], ...]

    @classmethod
    def from_context(cls, ctx: FormalContext) -> _IndexedDerivations:
        object_attributes = [set[int]() for _ in ctx.objects]
        attribute_objects = [set[int]() for _ in ctx.attributes]
        for object_index, attribute_index in ctx.incidence:
            object_attributes[object_index].add(attribute_index)
            attribute_objects[attribute_index].add(object_index)
        return cls(
            object_attributes=tuple(map(frozenset, object_attributes)),
            attribute_objects=tuple(map(frozenset, attribute_objects)),
        )

    def object_derivation(self, objects: frozenset[int]) -> frozenset[int]:
        attributes = set(range(len(self.attribute_objects)))
        for object_index in objects:
            attributes.intersection_update(self.object_attributes[object_index])
        return frozenset(attributes)

    def attribute_derivation(self, attributes: frozenset[int]) -> frozenset[int]:
        objects = set(range(len(self.object_attributes)))
        for attribute_index in attributes:
            objects.intersection_update(self.attribute_objects[attribute_index])
        return frozenset(objects)

    def attribute_closure(self, attributes: frozenset[int]) -> frozenset[int]:
        return self.object_derivation(self.attribute_derivation(attributes))


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
    derivations: _IndexedDerivations,
    current: frozenset[int],
    attribute_count: int,
) -> frozenset[int] | None:
    current_set = set(current)
    for index in range(attribute_count - 1, -1, -1):
        if index in current_set:
            current_set.discard(index)
            continue
        candidate = {attribute for attribute in current_set if attribute < index}
        candidate.add(index)
        closure = derivations.attribute_closure(frozenset(candidate))
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

    derivations = _IndexedDerivations.from_context(ctx)
    concepts: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    current: frozenset[int] | None = derivations.attribute_closure(frozenset())
    while current is not None:
        extent = derivations.attribute_derivation(current)
        concepts.append((tuple(sorted(extent)), tuple(sorted(current))))
        if len(concepts) > limit:
            raise ValueError(
                f"concept count exceeds maximum of {limit}; narrow the context "
                "or reduce the number of attributes"
            )
        current = _next_closure(derivations, current, len(ctx.attributes))
    return tuple(concepts)


def concept_family_size_capped(ctx: FormalContext, limit: int) -> int:
    derivations = _IndexedDerivations.from_context(ctx)
    current: frozenset[int] | None = derivations.attribute_closure(frozenset())
    count = 0
    while current is not None:
        count += 1
        if count > limit:
            return count
        current = _next_closure(derivations, current, len(ctx.attributes))
    return count
