"""Exact Duquenne--Guigues basis contracts (issue #2268)."""

from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.math.formal_concept_analysis import (
    CanonicalImplicationBasisResult,
    FormalContext,
    duquenne_guigues_basis,
    implication_closure,
)
from jacobian.math.formal_concept_analysis._models import (
    DuquenneGuiguesBasisRequest,
)
from jacobian.math.formal_concept_analysis._operations import (
    compute_duquenne_guigues_basis,
)
from jacobian.math.formal_concept_analysis.basis import (
    MAX_DG_ATTRIBUTES,
    MAX_DG_CANDIDATE_STATES,
    MAX_DG_LOGICAL_WORK,
    MAX_DG_RESULT_BYTES,
    _require_dg_canonical_carrier_fit,
    verify_canonical_implication_basis_result,
)
from jacobian.math.formal_concept_analysis.values import MAX_IMPLICATIONS


def _context(
    rows: tuple[tuple[int, ...], ...],
    attribute_count: int,
) -> FormalContext:
    return FormalContext(
        objects=tuple(f"g{index}" for index in range(len(rows))),
        attributes=tuple(f"m{index}" for index in range(attribute_count)),
        incidence=tuple(
            (object_index, attribute_index)
            for object_index, row in enumerate(rows)
            for attribute_index in row
        ),
    )


def _contranominal_context(attribute_count: int) -> FormalContext:
    return _context(
        tuple(
            tuple(attribute for attribute in range(attribute_count) if attribute != row)
            for row in range(attribute_count)
        ),
        attribute_count,
    )


def _mask(values: tuple[int, ...]) -> int:
    return sum(1 << value for value in values)


def _definition_oracle(
    context: FormalContext,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Enumerate by cardinality, independently of the producer's lectic scan."""

    attribute_count = len(context.attributes)
    incidences = set(context.incidence)

    def closure(subset: frozenset[int]) -> tuple[int, ...]:
        extent = tuple(
            object_index
            for object_index in range(len(context.objects))
            if all(
                (object_index, attribute_index) in incidences
                for attribute_index in subset
            )
        )
        return tuple(
            attribute_index
            for attribute_index in range(attribute_count)
            if all(
                (object_index, attribute_index) in incidences for object_index in extent
            )
        )

    pseudo_intents: list[tuple[frozenset[int], tuple[int, ...]]] = []
    for size in range(attribute_count + 1):
        for members in combinations(range(attribute_count), size):
            premise = frozenset(members)
            closed = closure(premise)
            if tuple(members) == closed:
                continue
            if all(
                not previous_premise < premise
                or set(previous_closure).issubset(premise)
                for previous_premise, previous_closure in pseudo_intents
            ):
                pseudo_intents.append((premise, closed))
    return tuple(
        (_mask(tuple(sorted(premise))), closure) for premise, closure in pseudo_intents
    )


def test_geometric_figures_textbook_context_has_three_pseudo_intents() -> None:
    # Ignatov, Introduction to FCA (Example 3 and Table 3): triangle,
    # right triangle, rectangle, and square with attributes a,b,c,d.
    context = FormalContext(
        objects=("triangle", "right_triangle", "rectangle", "square"),
        attributes=("a", "b", "c", "d"),
        incidence=(
            (0, 0),
            (0, 3),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
        ),
    )

    result = duquenne_guigues_basis(context)

    assert tuple(
        (row.candidate_state, row.premise, row.closure) for row in result.pseudo_intents
    ) == (
        (2, (1,), (1, 2)),
        (7, (0, 1, 2), (0, 1, 2, 3)),
        (12, (2, 3), (1, 2, 3)),
    )
    assert tuple(
        (implication.premise, implication.conclusion)
        for implication in result.basis.implications
    ) == (
        ((0, 1, 2), (3,)),
        ((1,), (2,)),
        ((2, 3), (1,)),
    )
    assert tuple(
        result.basis.implications[row.basis_implication_index].premise
        for row in result.pseudo_intents
    ) == tuple(row.premise for row in result.pseudo_intents)


def test_empty_basis_and_empty_premise_boundary_contexts() -> None:
    contranominal = _context(
        tuple(
            tuple(attribute for attribute in range(4) if attribute != object_index)
            for object_index in range(4)
        ),
        4,
    )
    assert duquenne_guigues_basis(contranominal).pseudo_intents == ()

    nonempty_empty_closure = _context(((0,), (0,)), 2)
    result = duquenne_guigues_basis(nonempty_empty_closure)
    assert tuple(row.premise for row in result.pseudo_intents) == ((),)
    assert tuple(
        (implication.premise, implication.conclusion)
        for implication in result.basis.implications
    ) == (((), (0,)),)
    assert result.closure_matrix[0].closure == (0,)


@pytest.mark.parametrize(
    "context",
    (
        _context(((0, 1, 2), (1, 2), (2,)), 3),
        _context(((0, 1), (0, 2), (0,), ()), 3),
    ),
)
def test_chain_and_diamond_contexts_replay_every_subset(
    context: FormalContext,
) -> None:
    result = duquenne_guigues_basis(context)

    assert len(result.closure_matrix) == 1 << len(context.attributes)
    for closure_row in result.closure_matrix:
        replay = implication_closure(
            result.basis,
            frozenset(closure_row.subset),
        )
        assert replay.closure == closure_row.closure


def test_every_binary_two_by_three_context_matches_definition_oracle() -> None:
    attribute_count = 3
    for incidence_mask in range(1 << (2 * attribute_count)):
        rows = tuple(
            tuple(
                attribute
                for attribute in range(attribute_count)
                if incidence_mask & (1 << (object_index * attribute_count + attribute))
            )
            for object_index in range(2)
        )
        context = _context(rows, attribute_count)

        result = duquenne_guigues_basis(context)

        actual = tuple(
            (row.candidate_state, row.closure) for row in result.pseudo_intents
        )
        assert set(actual) == set(_definition_oracle(context)), incidence_mask


def test_object_order_attribute_permutation_and_relabeling_are_coherent() -> None:
    original = _context(((0, 1), (1, 2), (0, 2)), 3)
    original_result = duquenne_guigues_basis(original)

    object_reordered = FormalContext(
        objects=tuple(reversed(original.objects)),
        attributes=original.attributes,
        incidence=tuple(
            sorted(
                (len(original.objects) - 1 - oi, ai) for oi, ai in original.incidence
            )
        ),
    )
    object_result = duquenne_guigues_basis(object_reordered)
    assert tuple(
        (row.premise, row.closure) for row in object_result.pseudo_intents
    ) == tuple((row.premise, row.closure) for row in original_result.pseudo_intents)

    permutation = (2, 0, 1)
    inverse = {old: new for new, old in enumerate(permutation)}
    permuted = FormalContext(
        objects=original.objects,
        attributes=tuple(original.attributes[old] for old in permutation),
        incidence=tuple(sorted((oi, inverse[ai]) for oi, ai in original.incidence)),
    )
    permuted_result = duquenne_guigues_basis(permuted)

    def by_label(
        result: CanonicalImplicationBasisResult,
    ) -> set[tuple[frozenset[str], frozenset[str]]]:
        return {
            (
                frozenset(result.context.attributes[index] for index in row.premise),
                frozenset(result.context.attributes[index] for index in row.closure),
            )
            for row in result.pseudo_intents
        }

    assert by_label(permuted_result) == by_label(original_result)

    relabeled = original.model_copy(update={"attributes": ("red", "green", "blue")})
    relabeled = FormalContext.model_validate(relabeled.model_dump())
    relabeled_result = duquenne_guigues_basis(relabeled)
    assert tuple(
        (row.premise, row.closure) for row in relabeled_result.pseudo_intents
    ) == tuple((row.premise, row.closure) for row in original_result.pseudo_intents)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("omit_pseudo_intent", "pseudo-intent"),
        ("invalid_premise", "pseudo-intent"),
        ("weaken_conclusion", "basis"),
        ("change_source", "closure matrix"),
        ("change_closure", "closure matrix"),
        ("change_work", "work accounting"),
    ),
)
def test_result_validator_rejects_corrupted_source_family_basis_matrix_and_work(
    mutation: str,
    message: str,
) -> None:
    result = duquenne_guigues_basis(_context(((0, 3), (0, 2), (1, 2), (1, 2, 3)), 4))
    payload = result.model_dump()
    if mutation == "omit_pseudo_intent":
        payload["pseudo_intents"] = payload["pseudo_intents"][:-1]
    elif mutation == "invalid_premise":
        payload["pseudo_intents"][0]["premise"] = [0]
    elif mutation == "weaken_conclusion":
        payload["basis"]["implications"][0]["conclusion"] = []
    elif mutation == "change_source":
        payload["context"]["incidence"] = payload["context"]["incidence"][:-1]
    elif mutation == "change_closure":
        payload["closure_matrix"][0]["closure"] = [0]
    else:
        payload["work"]["context_object_row_checks"] += 1
        payload["work"]["accounted_logical_work"] += 1

    with pytest.raises(ValueError):
        verify_canonical_implication_basis_result(
            CanonicalImplicationBasisResult.model_validate(payload)
        )


def test_coordinate_map_preserves_unrestricted_source_labels() -> None:
    context = FormalContext(
        objects=(" object labels remain source data ",),
        attributes=("", " attribute with surrounding whitespace "),
        incidence=((0, 0),),
    )

    result = duquenne_guigues_basis(context)

    assert result.context == context
    assert result.source_attribute_indices == (0, 1)
    assert result.basis.attributes == ("m0", "m1")


def test_nine_attribute_contranominal_context_admitted_with_empty_basis() -> None:
    context = _contranominal_context(9)

    result = duquenne_guigues_basis(context)

    assert result.work.candidate_states == 512
    assert len(result.closure_matrix) == 512
    for row in result.closure_matrix:
        assert row.subset == row.closure
    assert result.pseudo_intents == ()
    assert result.basis.implications == ()
    assert 0 < result.work.reserved_logical_work <= MAX_DG_LOGICAL_WORK
    assert result.work.reserved_result_bytes <= MAX_DG_RESULT_BYTES

    request = DuquenneGuiguesBasisRequest(context=context)
    replayed = compute_duquenne_guigues_basis(request)
    assert isinstance(replayed, CanonicalImplicationBasisResult)
    assert replayed.work.candidate_states == 512


def test_candidate_work_and_output_envelopes_at_boundary() -> None:
    boundary = _contranominal_context(MAX_DG_ATTRIBUTES)
    request = DuquenneGuiguesBasisRequest(context=boundary)
    result = compute_duquenne_guigues_basis(request)
    assert result.work.candidate_states == MAX_DG_CANDIDATE_STATES
    assert result.work.context_object_row_checks == (
        3 * MAX_DG_CANDIDATE_STATES * MAX_DG_ATTRIBUTES
    )
    assert 0 < result.work.reserved_logical_work < MAX_DG_LOGICAL_WORK
    assert result.work.reserved_result_bytes <= MAX_DG_RESULT_BYTES

    over_boundary = _contranominal_context(MAX_DG_ATTRIBUTES + 1)
    with pytest.raises(ValidationError):
        DuquenneGuiguesBasisRequest(context=over_boundary)
    with pytest.raises(ValueError, match="candidate-state"):
        duquenne_guigues_basis(over_boundary)


def test_exact_basis_beyond_canonical_implication_carrier_is_rejected() -> None:
    full_mask = (1 << MAX_DG_ATTRIBUTES) - 1
    wide_pairs = tuple((state, full_mask) for state in range(MAX_IMPLICATIONS + 1))
    with pytest.raises(ValueError, match="implications exceeds"):
        _require_dg_canonical_carrier_fit(wide_pairs)

    dense_pairs = tuple((state, (1 << 64) - 1) for state in range(MAX_IMPLICATIONS))
    with pytest.raises(ValueError, match="memberships exceeds"):
        _require_dg_canonical_carrier_fit(dense_pairs)


def test_work_accounting_includes_every_probe_pass_and_replay() -> None:
    context = _context(((0, 3), (0, 2), (1, 2), (1, 2, 3)), 4)
    result = duquenne_guigues_basis(context)

    attribute_count = len(context.attributes)
    states = len(result.closure_matrix)
    incidence = set(context.incidence)

    def extent_size(state: int) -> int:
        return sum(
            1
            for object_index in range(len(context.objects))
            if all(
                (object_index, attribute_index) in incidence
                for attribute_index in range(attribute_count)
                if state & (1 << attribute_index)
            )
        )

    row_intersections = sum(extent_size(state) for state in range(states))

    subset_comparisons = 0
    closure_comparisons = 0
    pseudo_intents: list[tuple[int, int]] = []
    for row in result.closure_matrix:
        premise_mask = row.candidate_state
        closure_mask = _mask(row.closure)
        if premise_mask == closure_mask:
            continue
        is_pseudo_intent = True
        for previous_state, previous_closure in pseudo_intents:
            subset_comparisons += 1
            if previous_state & premise_mask != previous_state:
                continue
            closure_comparisons += 1
            if previous_closure & premise_mask != previous_closure:
                is_pseudo_intent = False
                break
        if is_pseudo_intent:
            pseudo_intents.append((premise_mask, closure_mask))

    work = result.work

    assert work.context_closure_queries == 4 * states
    assert work.context_object_row_checks == (3 * states * len(context.objects))
    assert work.context_incidence_loads == 3 * len(context.incidence)
    assert work.context_row_intersections == 4 * row_intersections
    assert work.pseudo_intent_subset_comparisons == 3 * subset_comparisons
    assert work.pseudo_intent_closure_comparisons == 3 * closure_comparisons
    assert work.basis_closure_queries == 2 * states
    assert work.accounted_logical_work == (
        work.context_object_row_checks
        + work.context_incidence_loads
        + work.context_row_intersections
        + work.context_incidence_checks
        + work.pseudo_intent_subset_comparisons
        + work.pseudo_intent_closure_comparisons
        + 2 * work.basis_canonical_replay_work
        + work.closure_matrix_memberships
        + work.pseudo_intent_memberships
        + work.implication_memberships
    )
    assert work.reserved_logical_work >= work.accounted_logical_work


def test_native_and_catalog_invocations_report_identical_exact_work() -> None:
    context = _context(((0, 3), (0, 2), (1, 2), (1, 2, 3)), 4)

    native = duquenne_guigues_basis(context)
    via_request = compute_duquenne_guigues_basis(
        DuquenneGuiguesBasisRequest(context=context)
    )

    assert native == via_request
    assert native.work.context_closure_queries == 4 * len(native.closure_matrix)


def test_result_byte_reservation_accepts_its_last_byte_and_rejects_the_next() -> None:
    def accepts(label_length: int) -> bool:
        context = FormalContext(
            objects=("g" * label_length,),
            attributes=("m",),
            incidence=(),
        )
        try:
            DuquenneGuiguesBasisRequest(context=context)
        except ValidationError:
            return False
        return True

    low = 1
    high = MAX_DG_RESULT_BYTES + 1
    while low + 1 < high:
        middle = (low + high) // 2
        if accepts(middle):
            low = middle
        else:
            high = middle

    assert accepts(low)
    assert not accepts(low + 1)


def test_request_schema_discloses_state_work_and_output_envelopes() -> None:
    schema = DuquenneGuiguesBasisRequest.model_json_schema()
    description = schema["description"]

    assert f"{MAX_DG_CANDIDATE_STATES} states" in description
    assert f"{MAX_DG_ATTRIBUTES} attributes" in description
    assert f"{MAX_IMPLICATIONS}-implication" in description
    assert f"{MAX_DG_LOGICAL_WORK:,}" in description
    assert f"{MAX_DG_RESULT_BYTES:,}" in description


def test_adapter_returns_the_source_bound_native_value() -> None:
    context = _context(((0,), (1,)), 2)
    request = DuquenneGuiguesBasisRequest(context=context)

    result = compute_duquenne_guigues_basis(request)

    assert isinstance(result, CanonicalImplicationBasisResult)
    assert result.context == context
