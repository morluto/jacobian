"""Exact native kernels for formal concept analysis."""

from __future__ import annotations

from jacobian.canonical import encode_strict_json

from ._models import MAX_CONCEPTS
from .basis import (
    CanonicalImplicationBasisResult,
    DGBasisClosureRow,
    PseudoIntent,
    _basis_attribute_labels,
    _duquenne_guigues_preflight,
    _enumerate_dg_masks,
    _subset_for_state,
)
from .values import (
    AttributeImplication,
    FiniteAttributeImplicationSystem,
    FormalContext,
    ImplicationClosureResult,
    ImplicationClosureWork,
    ImplicationDerivation,
)

__all__ = [
    "MAX_CONCEPTS",
    "attribute_closure",
    "attribute_derivation",
    "concept_family_size_capped",
    "concept_from_attributes",
    "concept_from_objects",
    "concept_lattice",
    "duquenne_guigues_basis",
    "enumerate_concepts",
    "implication_closure",
    "object_closure",
    "object_derivation",
]


def _require_implication_seed(
    system: FiniteAttributeImplicationSystem,
    seed: frozenset[int],
) -> None:
    for attribute in seed:
        if type(attribute) is not int:
            raise TypeError("implication seed attributes must be integers")
        if not 0 <= attribute < len(system.attributes):
            raise ValueError(
                "implication seed attribute is outside the declared carrier"
            )


def implication_closure(
    system: FiniteAttributeImplicationSystem,
    seed: frozenset[int],
) -> ImplicationClosureResult:
    """Return the exact least superset of ``seed`` closed under ``system``.

    Each round evaluates every canonical implication against the closure at the
    start of that round.  If several enabled implications first derive the same
    attribute, the first implication in canonical row order owns its lineage.
    A final nonproductive scan establishes that the returned set is closed.
    """

    _require_implication_seed(system, seed)
    closure = set(seed)
    lineage: list[ImplicationDerivation] = []
    productive_rounds = 0
    implication_checks = 0
    membership_checks = 0

    while True:
        first_sources: dict[int, int] = {}
        for implication_index, implication in enumerate(system.implications):
            implication_checks += 1
            membership_checks += len(implication.premise)
            if not set(implication.premise).issubset(closure):
                continue
            membership_checks += len(implication.conclusion)
            for attribute in implication.conclusion:
                if attribute not in closure:
                    first_sources.setdefault(attribute, implication_index)

        if not first_sources:
            break

        productive_rounds += 1
        for attribute in sorted(first_sources):
            lineage.append(
                ImplicationDerivation(
                    attribute=attribute,
                    implication_index=first_sources[attribute],
                    activation_round=productive_rounds,
                )
            )
        closure.update(first_sources)

    canonical_replay_work = implication_checks + membership_checks
    return ImplicationClosureResult._from_kernel(
        system=system,
        seed=tuple(sorted(seed)),
        closure=tuple(sorted(closure)),
        added=tuple(sorted(closure - set(seed))),
        lineage=tuple(lineage),
        work=ImplicationClosureWork(
            productive_rounds=productive_rounds,
            canonical_implication_checks=implication_checks,
            canonical_membership_checks=membership_checks,
            canonical_replay_work=canonical_replay_work,
        ),
    )


def _enumerate_dg_basis(
    context: FormalContext,
) -> tuple[
    tuple[DGBasisClosureRow, ...],
    tuple[tuple[int, int], ...],
    int,
    int,
    int,
]:
    """Enumerate all closures and pseudo-intents with bounded integer bitsets."""

    attribute_count = len(context.attributes)
    (
        closure_masks,
        pseudo_intent_masks,
        subset_comparisons,
        closure_comparisons,
        (row_intersections),
    ) = _enumerate_dg_masks(context)
    closure_rows = tuple(
        DGBasisClosureRow(
            candidate_state=state,
            subset=_subset_for_state(state, attribute_count),
            closure=_subset_for_state(closure_mask, attribute_count),
        )
        for state, closure_mask in enumerate(closure_masks)
    )
    return (
        closure_rows,
        pseudo_intent_masks,
        subset_comparisons,
        closure_comparisons,
        row_intersections,
    )


def _result_payload_with_exact_wire_bytes(
    payload: dict[str, object],
) -> dict[str, object]:
    work = payload["work"]
    if not isinstance(work, dict):
        raise TypeError("internal DG-basis work payload must be an object")
    serialized_result_bytes = 1
    for _ in range(4):
        work["serialized_result_bytes"] = serialized_result_bytes
        measured = len(encode_strict_json(payload))
        if measured == serialized_result_bytes:
            return payload
        serialized_result_bytes = measured
    raise RuntimeError("serialized DG-basis result size did not reach a fixed point")


def duquenne_guigues_basis(
    context: FormalContext,
) -> CanonicalImplicationBasisResult:
    """Return every pseudo-intent and the exact canonical implication basis.

    Candidate states are visited in binary lectic order, where the largest
    differing source attribute decides the order.  Numeric bit-mask order is a
    linear extension of subset inclusion, so every proper-subset pseudo-intent
    needed by the recursive definition has already been considered.
    """

    states, reserved_logical_work, reserved_result_bytes = _duquenne_guigues_preflight(
        context
    )
    (
        closure_matrix,
        pseudo_intent_masks,
        subset_comparisons,
        closure_comparisons,
        row_intersections,
    ) = _enumerate_dg_basis(context)
    attribute_count = len(context.attributes)
    implications = tuple(
        AttributeImplication(
            premise=_subset_for_state(state, attribute_count),
            conclusion=_subset_for_state(closure & ~state, attribute_count),
        )
        for state, closure in pseudo_intent_masks
    )
    basis = FiniteAttributeImplicationSystem(
        attributes=_basis_attribute_labels(attribute_count),
        implications=implications,
    )
    implication_indices = {
        implication.premise: index
        for index, implication in enumerate(basis.implications)
    }
    pseudo_intents = tuple(
        PseudoIntent(
            candidate_state=state,
            premise=_subset_for_state(state, attribute_count),
            closure=_subset_for_state(closure, attribute_count),
            basis_implication_index=implication_indices[
                _subset_for_state(state, attribute_count)
            ],
        )
        for state, closure in pseudo_intent_masks
    )

    basis_replay_work = 0
    for closure_row in closure_matrix:
        replay = implication_closure(basis, frozenset(closure_row.subset))
        if replay.closure != closure_row.closure:
            raise RuntimeError(
                "constructed canonical basis failed source closure equivalence"
            )
        basis_replay_work += replay.work.canonical_replay_work

    closure_matrix_memberships = sum(
        len(row.subset) + len(row.closure) for row in closure_matrix
    )
    pseudo_intent_memberships = sum(
        len(row.premise) + len(row.closure) for row in pseudo_intents
    )
    incidence_checks = len(context.incidence) * (
        attribute_count * states // 2 + row_intersections
    )
    implication_memberships = basis.total_memberships
    # Exact accounting covers the three kernel-executed closure-matrix passes
    # (producer preflight, producer enumeration, result-validation preflight),
    # the result validator's independent per-state reconstruction, and the
    # recursive pseudo-intent replays those passes run.  Request-model
    # admission probing precedes the kernel and stays outside the reported
    # result, so native and catalog invocations report identical counts.
    accounted_logical_work = (
        3 * states * len(context.objects)
        + 3 * len(context.incidence)
        + 4 * row_intersections
        + incidence_checks
        + 3 * subset_comparisons
        + 3 * closure_comparisons
        + 4 * basis_replay_work
        + closure_matrix_memberships
        + pseudo_intent_memberships
        + implication_memberships
    )
    payload: dict[str, object] = {
        "context": context.model_dump(mode="json"),
        "source_attribute_indices": list(range(attribute_count)),
        "lectic_order": "BINARY_LECTIC_BY_MAXIMUM_DIFFERENCE",
        "closure_matrix": [row.model_dump(mode="json") for row in closure_matrix],
        "pseudo_intents": [row.model_dump(mode="json") for row in pseudo_intents],
        "basis": basis.model_dump(mode="json"),
        "work": {
            "candidate_states": states,
            "context_closure_queries": 4 * states,
            "context_object_row_checks": 3 * states * len(context.objects),
            "context_incidence_loads": 3 * len(context.incidence),
            "context_row_intersections": 4 * row_intersections,
            "context_incidence_checks": incidence_checks,
            "pseudo_intent_subset_comparisons": 3 * subset_comparisons,
            "pseudo_intent_closure_comparisons": 3 * closure_comparisons,
            "basis_closure_queries": 2 * states,
            "basis_canonical_replay_work": 2 * basis_replay_work,
            "closure_matrix_memberships": closure_matrix_memberships,
            "pseudo_intent_memberships": pseudo_intent_memberships,
            "implication_count": len(basis.implications),
            "implication_memberships": implication_memberships,
            "accounted_logical_work": accounted_logical_work,
            "reserved_logical_work": reserved_logical_work,
            "reserved_result_bytes": reserved_result_bytes,
            "serialized_result_bytes": 1,
        },
    }
    return CanonicalImplicationBasisResult._from_kernel(
        _result_payload_with_exact_wire_bytes(payload)
    )


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

    The algorithm enumerates closed attribute intents in lectic order,
    starting from ``attribute_closure(ctx, frozenset())`` because in a
    general context the empty attribute set need not be closed; the least
    closed intent is ``cl(∅) = ∅''``.  Each step requires O(n) derivation
    operations, so the total cost is proportional to the number of concepts
    times n, not to 2^n.
    """
    n = len(ctx.attributes)
    concepts: list[dict[str, frozenset[int]]] = []

    current: frozenset[int] | None = attribute_closure(ctx, frozenset())
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


def concept_family_size_capped(ctx: FormalContext, limit: int) -> int:
    """Return the exact concept-family size, aborting once it exceeds ``limit``.

    Walks the same NextClosure order as :func:`enumerate_concepts` but keeps
    only a counter, so the work is bounded by ``limit + 1`` closure steps
    regardless of the true family size.  Admission uses this to decide
    overflow exactly for contexts whose worst case alone cannot prove that
    the family fits the declared budget.
    """
    n = len(ctx.attributes)
    count = 0
    current: frozenset[int] | None = attribute_closure(ctx, frozenset())
    while current is not None:
        count += 1
        if count > limit:
            return count
        current = _next_closure(ctx, current, n)
    return count


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
    return _concept_lattice_from_concepts(enumerate_concepts(ctx))


def _concept_lattice_from_canonical_concepts(
    concepts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
) -> dict[str, object]:
    """Derive one lattice from an already admitted canonical concept family."""

    return _concept_lattice_from_concepts(
        [
            {"extent": frozenset(extent), "intent": frozenset(intent)}
            for extent, intent in concepts
        ]
    )


def _concept_lattice_from_concepts(
    concepts: list[dict[str, frozenset[int]]],
) -> dict[str, object]:
    """Derive one lattice from a complete exact concept family."""

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
