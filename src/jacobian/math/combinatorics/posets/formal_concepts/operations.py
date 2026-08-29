"""Exact native kernels for formal concept analysis."""

from __future__ import annotations

from typing import NoReturn, TypedDict

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError

from . import _concepts
from ._models import (
    MAX_CONCEPTS,
    ClosureResult,
    ConceptLatticeResult,
    ConceptResult,
)
from .basis import (
    CanonicalImplicationBasisResult,
    DGBasisClosureRow,
    PseudoIntent,
    _admit_duquenne_guigues_basis,
    _basis_attribute_labels,
    _DGBasisAdmissionPlan,
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


class _Concept(TypedDict):
    extent: frozenset[int]
    intent: frozenset[int]


def _domain_error(location: tuple[str, ...], code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code=f"formal_concept_analysis.{code}",
        message=message,
    )


def _require_implication_seed(
    system: FiniteAttributeImplicationSystem,
    seed: frozenset[int],
) -> None:
    for attribute in seed:
        if type(attribute) is not int:
            _domain_error(
                ("seed",),
                "seed_attribute_must_be_integer",
                "implication seed attributes must be integers",
            )
        if not 0 <= attribute < len(system.attributes):
            _domain_error(
                ("seed",),
                "seed_attribute_out_of_range",
                "implication seed attribute is outside the declared carrier",
            )


def _admit_subset(context: FormalContext, subset: frozenset[int], *, side: str) -> None:
    size = len(context.objects) if side == "object" else len(context.attributes)
    for index in subset:
        if type(index) is not int:
            _domain_error(
                (side,),
                "subset_index_must_be_integer",
                f"{side} subset indices must be integers",
            )
        if not 0 <= index < size:
            _domain_error(
                (side,),
                "subset_index_out_of_range",
                f"{side} subset index out of range",
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

    total_logical_work = implication_checks + membership_checks
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
            total_logical_work=total_logical_work,
        ),
    )


def _closure_rows_from_plan(
    context: FormalContext,
    plan: _DGBasisAdmissionPlan,
) -> tuple[DGBasisClosureRow, ...]:
    """Materialize canonical rows without repeating semantic admission."""

    attribute_count = len(context.attributes)
    return tuple(
        DGBasisClosureRow(
            candidate_state=state,
            subset=_subset_for_state(state, attribute_count),
            closure=_subset_for_state(closure_mask, attribute_count),
        )
        for state, closure_mask in enumerate(plan.closure_masks)
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

    try:
        plan = _admit_duquenne_guigues_basis(context)
    except ValueError as exc:
        _domain_error(("context",), "basis_admission_failed", str(exc))
    closure_matrix = _closure_rows_from_plan(context, plan)
    attribute_count = len(context.attributes)
    implications = tuple(
        AttributeImplication(
            premise=_subset_for_state(state, attribute_count),
            conclusion=_subset_for_state(closure & ~state, attribute_count),
        )
        for state, closure in plan.pseudo_intent_pairs
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
        for state, closure in plan.pseudo_intent_pairs
    )

    basis_closure_work = 0
    for closure_row in closure_matrix:
        closure_result = implication_closure(basis, frozenset(closure_row.subset))
        if closure_result.closure != closure_row.closure:
            raise RuntimeError(
                "constructed canonical basis failed source closure equivalence"
            )
        basis_closure_work += closure_result.work.total_logical_work

    closure_matrix_memberships = sum(
        len(row.subset) + len(row.closure) for row in closure_matrix
    )
    pseudo_intent_memberships = sum(
        len(row.premise) + len(row.closure) for row in pseudo_intents
    )
    implication_memberships = basis.total_memberships
    accounted_logical_work = (
        plan.states * len(context.objects)
        + len(context.incidence)
        + plan.row_intersections
        + plan.subset_comparisons
        + plan.closure_comparisons
        + basis_closure_work
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
            "candidate_states": plan.states,
            "context_closure_queries": plan.states,
            "context_object_row_checks": plan.states * len(context.objects),
            "context_incidence_loads": len(context.incidence),
            "context_row_intersections": plan.row_intersections,
            "pseudo_intent_subset_comparisons": plan.subset_comparisons,
            "pseudo_intent_closure_comparisons": plan.closure_comparisons,
            "basis_closure_queries": plan.states,
            "basis_closure_work": basis_closure_work,
            "closure_matrix_memberships": closure_matrix_memberships,
            "pseudo_intent_memberships": pseudo_intent_memberships,
            "implication_count": len(basis.implications),
            "implication_memberships": implication_memberships,
            "accounted_logical_work": accounted_logical_work,
            "reserved_logical_work": plan.reserved_logical_work,
            "reserved_result_bytes": plan.reserved_result_bytes,
            "serialized_result_bytes": 1,
        },
    }
    return CanonicalImplicationBasisResult._from_payload(
        _result_payload_with_exact_wire_bytes(payload)
    )


def object_derivation(ctx: FormalContext, objects: frozenset[int]) -> frozenset[int]:
    """Return A' = {m in M : every g in A has attribute m}.

    Under standard FCA semantics, the derivation of the empty object set is
    every attribute.
    """
    _admit_subset(ctx, objects, side="object")
    return _concepts.object_derivation(ctx, objects)


def attribute_derivation(
    ctx: FormalContext, attributes: frozenset[int]
) -> frozenset[int]:
    """Return B' = {g in G : every m in B is possessed by g}.

    Under standard FCA semantics, the derivation of the empty attribute set is
    every object.
    """
    _admit_subset(ctx, attributes, side="attribute")
    return _concepts.attribute_derivation(ctx, attributes)


def object_closure(ctx: FormalContext, objects: frozenset[int]) -> frozenset[int]:
    """Return A'' = (A')'."""
    _admit_subset(ctx, objects, side="object")
    return _concepts.attribute_derivation(
        ctx, _concepts.object_derivation(ctx, objects)
    )


def object_closure_result(ctx: FormalContext, objects: frozenset[int]) -> ClosureResult:
    """Return object closure data from one admitted canonical subset."""
    _admit_subset(ctx, objects, side="object")
    derived = _concepts.object_derivation(ctx, objects)
    closure = _concepts.attribute_derivation(ctx, derived)
    return ClosureResult(
        closure=tuple(sorted(closure)),
        derived=tuple(sorted(derived)),
        added=tuple(sorted(closure - objects)),
        is_closed=closure == objects,
    )


def attribute_closure(ctx: FormalContext, attributes: frozenset[int]) -> frozenset[int]:
    """Return B'' = (B')'."""
    _admit_subset(ctx, attributes, side="attribute")
    return _concepts.object_derivation(
        ctx, _concepts.attribute_derivation(ctx, attributes)
    )


def attribute_closure_result(
    ctx: FormalContext, attributes: frozenset[int]
) -> ClosureResult:
    """Return attribute closure data from one admitted canonical subset."""
    _admit_subset(ctx, attributes, side="attribute")
    derived = _concepts.attribute_derivation(ctx, attributes)
    closure = _concepts.object_derivation(ctx, derived)
    return ClosureResult(
        closure=tuple(sorted(closure)),
        derived=tuple(sorted(derived)),
        added=tuple(sorted(closure - attributes)),
        is_closed=closure == attributes,
    )


def concept_from_objects(ctx: FormalContext, objects: frozenset[int]) -> ConceptResult:
    """Return the unique concept (A'', A')."""
    _admit_subset(ctx, objects, side="object")
    intent = _concepts.object_derivation(ctx, objects)
    extent = _concepts.attribute_derivation(ctx, intent)
    return ConceptResult(
        extent=tuple(sorted(extent)),
        intent=tuple(sorted(intent)),
    )


def concept_from_attributes(
    ctx: FormalContext, attributes: frozenset[int]
) -> ConceptResult:
    """Return the unique concept (B', B'')."""
    _admit_subset(ctx, attributes, side="attribute")
    extent = _concepts.attribute_derivation(ctx, attributes)
    intent = _concepts.object_derivation(ctx, extent)
    return ConceptResult(
        extent=tuple(sorted(extent)),
        intent=tuple(sorted(intent)),
    )


def enumerate_concepts(ctx: FormalContext) -> list[_Concept]:
    """Return every formal concept exactly once using Ganter's NextClosure
    algorithm over the declared attribute order.

    The algorithm enumerates closed attribute intents in lectic order,
    starting from ``attribute_closure(ctx, frozenset())`` because in a
    general context the empty attribute set need not be closed; the least
    closed intent is ``cl(∅) = ∅''``.  Each step requires O(n) derivation
    operations, so the total cost is proportional to the number of concepts
    times n, not to 2^n.
    """
    try:
        pairs = _concepts.enumerate_concept_pairs(ctx, limit=MAX_CONCEPTS)
    except ValueError as exc:
        _domain_error(("context",), "concept_enumeration_admission_failed", str(exc))
    return [
        {"extent": frozenset(extent), "intent": frozenset(intent)}
        for extent, intent in pairs
    ]


def concept_family_size_capped(ctx: FormalContext, limit: int) -> int:
    """Return the exact concept-family size, aborting once it exceeds ``limit``.

    Walks the same NextClosure order as :func:`enumerate_concepts` but keeps
    only a counter, so the work is bounded by ``limit + 1`` closure steps
    regardless of the true family size.  Admission uses this to decide
    overflow exactly for contexts whose worst case alone cannot prove that
    the family fits the declared budget.
    """
    return _concepts.concept_family_size_capped(ctx, limit)


def _inclusion_order(
    concepts: list[_Concept],
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
) -> ConceptLatticeResult:
    """Return the concept lattice: concepts, partial order by extent inclusion,
    cover relation, top and bottom concepts."""
    return _concept_lattice_from_concepts(enumerate_concepts(ctx))


def _concept_lattice_from_canonical_concepts(
    concepts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
) -> ConceptLatticeResult:
    """Derive one lattice from an already admitted canonical concept family."""

    return _concept_lattice_from_concepts(
        [
            {"extent": frozenset(extent), "intent": frozenset(intent)}
            for extent, intent in concepts
        ]
    )


def _concept_lattice_from_concepts(
    concepts: list[_Concept],
) -> ConceptLatticeResult:
    """Derive one lattice from a complete exact concept family."""

    n = len(concepts)
    order = _inclusion_order(concepts)
    covers = _cover_relation(order, n)
    if n == 0:
        return ConceptLatticeResult(
            concepts=(), order=(), covers=(), top=None, bottom=None
        )
    bottom = 0
    top = 0
    for i in range(n):
        if concepts[i]["extent"] < concepts[bottom]["extent"]:
            bottom = i
        if concepts[i]["extent"] > concepts[top]["extent"]:
            top = i
    return ConceptLatticeResult(
        concepts=tuple(
            (tuple(sorted(concept["extent"])), tuple(sorted(concept["intent"])))
            for concept in concepts
        ),
        order=tuple(order),
        covers=tuple(covers),
        top=top,
        bottom=bottom,
    )
