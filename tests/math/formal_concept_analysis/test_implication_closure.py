"""Exact finite attribute-implication closure contracts (issue #2267)."""

from __future__ import annotations

from itertools import combinations, islice

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.math.formal_concept_analysis import (
    AttributeImplication,
    FiniteAttributeImplicationSystem,
    implication_closure,
)
from jacobian.math.formal_concept_analysis._models import ImplicationClosureRequest
from jacobian.math.formal_concept_analysis._operations import (
    compute_implication_closure,
)
from jacobian.math.formal_concept_analysis.values import (
    MAX_ATTRIBUTES,
    MAX_IMPLICATION_CLOSURE_RESULT_BYTES,
    MAX_IMPLICATION_MEMBERSHIPS,
    MAX_IMPLICATIONS,
    ImplicationClosureResult,
)


def _chain_system() -> FiniteAttributeImplicationSystem:
    return FiniteAttributeImplicationSystem(
        attributes=("a", "b", "c", "unreachable"),
        implications=(
            AttributeImplication(premise=(0,), conclusion=(1,)),
            AttributeImplication(premise=(1,), conclusion=(2,)),
            AttributeImplication(premise=(3,), conclusion=(0,)),
        ),
    )


def _brute_force_closure(
    system: FiniteAttributeImplicationSystem,
    seed: frozenset[int],
) -> tuple[int, ...]:
    carrier = range(len(system.attributes))
    closed_supersets: list[set[int]] = []
    for size in range(len(seed), len(system.attributes) + 1):
        for members in combinations(carrier, size):
            candidate = set(members)
            if not seed.issubset(candidate):
                continue
            if all(
                not set(rule.premise).issubset(candidate)
                or set(rule.conclusion).issubset(candidate)
                for rule in system.implications
            ):
                closed_supersets.append(candidate)
    closure = set.intersection(*closed_supersets)
    return tuple(sorted(closure))


def test_multi_round_closure_has_replayable_first_lineage() -> None:
    result = implication_closure(_chain_system(), frozenset({0}))

    assert result.closure == (0, 1, 2)
    assert result.added == (1, 2)
    assert tuple(
        (step.attribute, step.implication_index, step.activation_round)
        for step in result.lineage
    ) == ((1, 0, 1), (2, 1, 2))
    assert result.work.productive_rounds == 2
    assert result.work.canonical_replay_work == (
        result.work.canonical_implication_checks
        + result.work.canonical_membership_checks
    )


def test_empty_carrier_and_empty_system_have_empty_exact_closure() -> None:
    system = FiniteAttributeImplicationSystem(attributes=(), implications=())

    result = implication_closure(system, frozenset())

    assert result.closure == ()
    assert result.added == ()
    assert result.lineage == ()
    assert result.work.canonical_replay_work == 0


def test_empty_premise_fires_and_empty_conclusion_is_inert() -> None:
    system = FiniteAttributeImplicationSystem(
        attributes=("a", "b"),
        implications=(
            AttributeImplication(premise=(), conclusion=(0,)),
            AttributeImplication(premise=(0,), conclusion=(0,)),
        ),
    )

    result = implication_closure(system, frozenset())

    assert result.closure == (0,)
    assert result.lineage[0].activation_round == 1
    assert system.implications[1].conclusion == ()


def test_cyclic_rules_do_not_create_unjustified_attributes() -> None:
    system = FiniteAttributeImplicationSystem(
        attributes=("a", "b"),
        implications=(
            AttributeImplication(premise=(0,), conclusion=(1,)),
            AttributeImplication(premise=(1,), conclusion=(0,)),
        ),
    )

    assert implication_closure(system, frozenset()).closure == ()
    assert implication_closure(system, frozenset({0})).closure == (0, 1)


def test_rule_order_and_member_order_are_canonicalized() -> None:
    left = FiniteAttributeImplicationSystem(
        attributes=("a", "b", "c"),
        implications=(
            AttributeImplication(premise=(1,), conclusion=(2,)),
            AttributeImplication(premise=(0,), conclusion=(1, 0)),
        ),
    )
    right = FiniteAttributeImplicationSystem(
        attributes=("a", "b", "c"),
        implications=(
            AttributeImplication(premise=(0,), conclusion=(0, 1)),
            AttributeImplication(premise=(1,), conclusion=(2,)),
        ),
    )

    assert left == right
    assert implication_closure(left, frozenset({0})) == implication_closure(
        right, frozenset({0})
    )


def test_coherent_attribute_relabeling_preserves_indexed_closure() -> None:
    relabeled = _chain_system().model_copy(update={"attributes": ("x", "y", "z", "w")})
    relabeled = FiniteAttributeImplicationSystem.model_validate(relabeled.model_dump())

    assert implication_closure(relabeled, frozenset({0})).closure == (0, 1, 2)


def test_exhaustive_small_closures_match_intersection_of_closed_supersets() -> None:
    systems = (
        FiniteAttributeImplicationSystem(
            attributes=("a", "b", "c"),
            implications=(
                AttributeImplication(premise=(), conclusion=(0,)),
                AttributeImplication(premise=(0,), conclusion=(1,)),
                AttributeImplication(premise=(1,), conclusion=(2,)),
            ),
        ),
        FiniteAttributeImplicationSystem(
            attributes=("a", "b", "c"),
            implications=(
                AttributeImplication(premise=(0, 1), conclusion=(2,)),
                AttributeImplication(premise=(2,), conclusion=(0,)),
            ),
        ),
    )
    for system in systems:
        for size in range(4):
            for members in combinations(range(3), size):
                seed = frozenset(members)
                assert implication_closure(system, seed).closure == (
                    _brute_force_closure(system, seed)
                )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("closure", [0, 1], "added attributes"),
        ("closure", [0, 1, 2, 3], "added attributes"),
        (
            "lineage",
            [
                {"attribute": 1, "implication_index": 1, "activation_round": 1},
                {"attribute": 2, "implication_index": 1, "activation_round": 2},
            ],
            "lineage",
        ),
    ],
)
def test_result_validation_rejects_forged_conclusions(
    field: str,
    replacement: object,
    message: str,
) -> None:
    result = implication_closure(_chain_system(), frozenset({0}))
    payload = result.model_dump()
    payload[field] = replacement

    with pytest.raises(ValidationError):
        ImplicationClosureResult.model_validate(payload)


def test_closed_nonleast_superset_is_rejected_by_lineage_replay() -> None:
    result = implication_closure(_chain_system(), frozenset({0}))
    payload = result.model_dump()
    payload["closure"] = [0, 1, 2, 3]
    payload["added"] = [1, 2, 3]
    payload["lineage"] = [
        {"attribute": 1, "implication_index": 0, "activation_round": 1},
        {"attribute": 3, "implication_index": 2, "activation_round": 1},
        {"attribute": 2, "implication_index": 1, "activation_round": 2},
    ]

    with pytest.raises(ValidationError):
        ImplicationClosureResult.model_validate(payload)


def test_coherently_omitted_consequence_is_rejected_as_not_closed() -> None:
    result = implication_closure(_chain_system(), frozenset({0}))
    payload = result.model_dump()
    payload["closure"] = [0, 1]
    payload["added"] = [1]
    payload["lineage"] = payload["lineage"][:1]
    payload["work"] = {
        "productive_rounds": 1,
        "canonical_implication_checks": 6,
        "canonical_membership_checks": 9,
        "canonical_replay_work": 15,
    }

    with pytest.raises(ValidationError):
        ImplicationClosureResult.model_validate(payload)


def test_lineage_cannot_delay_an_already_enabled_derivation() -> None:
    system = FiniteAttributeImplicationSystem(
        attributes=("a", "b", "c"),
        implications=(
            AttributeImplication(premise=(0,), conclusion=(1,)),
            AttributeImplication(premise=(0,), conclusion=(2,)),
        ),
    )
    result = implication_closure(system, frozenset({0}))
    payload = result.model_dump()
    payload["lineage"][1]["activation_round"] = 2
    payload["work"]["productive_rounds"] = 2

    with pytest.raises(ValidationError):
        ImplicationClosureResult.model_validate(payload)


def test_work_cannot_append_a_nonproductive_round() -> None:
    result = implication_closure(_chain_system(), frozenset({0}))
    payload = result.model_dump()
    payload["work"]["productive_rounds"] += 1

    with pytest.raises(ValidationError):
        ImplicationClosureResult.model_validate(payload)


def test_result_validation_rejects_source_and_work_mutations() -> None:
    result = implication_closure(_chain_system(), frozenset({0}))

    seed_payload = result.model_dump()
    seed_payload["seed"] = [0, 3]
    with pytest.raises(ValidationError):
        ImplicationClosureResult.model_validate(seed_payload)

    system_payload = result.model_dump()
    system_payload["system"]["implications"][0]["conclusion"] = [3]
    with pytest.raises(ValidationError):
        ImplicationClosureResult.model_validate(system_payload)

    work_payload = result.model_dump()
    work_payload["work"]["canonical_implication_checks"] += 1
    work_payload["work"]["canonical_replay_work"] += 1
    with pytest.raises(ValidationError):
        ImplicationClosureResult.model_validate(work_payload)


def test_request_rejects_duplicate_or_foreign_seed_indices() -> None:
    system = _chain_system()
    with pytest.raises(ValidationError):
        ImplicationClosureRequest(system=system, seed=(0, 0))
    with pytest.raises(ValidationError):
        ImplicationClosureRequest(system=system, seed=(4,))


def test_request_schema_discloses_aggregate_system_constraints() -> None:
    schema = ImplicationClosureRequest.model_json_schema()
    system_schema = schema["$defs"]["FiniteAttributeImplicationSystem"]
    implication_description = system_schema["properties"]["implications"]["description"]

    assert "4,096 aggregate" in implication_description
    assert "duplicate rows after normalization" in implication_description


def test_system_rejects_duplicate_rows_after_conclusion_normalization() -> None:
    with pytest.raises(ValidationError):
        FiniteAttributeImplicationSystem(
            attributes=("a", "b"),
            implications=(
                AttributeImplication(premise=(0,), conclusion=(0, 1)),
                AttributeImplication(premise=(0,), conclusion=(1,)),
            ),
        )


def test_system_rejects_foreign_indices_and_non_strict_integer_indices() -> None:
    with pytest.raises(ValidationError):
        FiniteAttributeImplicationSystem(
            attributes=("a",),
            implications=(AttributeImplication(premise=(1,), conclusion=()),),
        )
    with pytest.raises(ValidationError):
        AttributeImplication(premise=(True,), conclusion=())
    with pytest.raises(ValidationError):
        FiniteAttributeImplicationSystem(attributes=("\ud800",))


def test_zero_work_carrier_beyond_the_formal_context_cap_is_admitted() -> None:
    attributes = tuple(f"a{i}" for i in range(200))
    system = FiniteAttributeImplicationSystem(attributes=attributes, implications=())

    result = implication_closure(system, frozenset())

    assert system.attributes == attributes
    assert result.closure == ()
    assert result.added == ()
    assert result.lineage == ()
    assert result.work.productive_rounds == 0
    assert result.work.canonical_replay_work == 0
    assert ImplicationClosureRequest(system=system).seed == ()


def test_large_carrier_chain_closes_exactly_within_the_work_budget() -> None:
    attributes = tuple(f"a{i}" for i in range(100))
    system = FiniteAttributeImplicationSystem(
        attributes=attributes,
        implications=tuple(
            AttributeImplication(premise=(i,), conclusion=(i + 1,)) for i in range(99)
        ),
    )

    result = implication_closure(system, frozenset({0}))

    assert result.closure == tuple(range(100))
    assert result.added == tuple(range(1, 100))
    assert [
        (step.attribute, step.implication_index, step.activation_round)
        for step in result.lineage
    ] == [(i, i - 1, i) for i in range(1, 100)]
    assert result.work.productive_rounds == 99


def test_genuine_work_overload_is_rejected_by_the_work_budget() -> None:
    attributes = tuple(f"a{i}" for i in range(300))
    rules = tuple(
        AttributeImplication(premise=premise, conclusion=())
        for premise in islice(combinations(range(300), 4), MAX_IMPLICATIONS)
    )
    assert sum(len(rule.premise) for rule in rules) <= MAX_IMPLICATION_MEMBERSHIPS

    with pytest.raises(ValidationError):
        FiniteAttributeImplicationSystem(attributes=attributes, implications=rules)


def test_wide_long_label_carrier_is_rejected_by_the_result_budget() -> None:
    attributes = tuple(f"{index:03d}" + "x" * 22 for index in range(700))

    with pytest.raises(ValidationError):
        FiniteAttributeImplicationSystem(attributes=attributes)


def test_implication_count_boundary() -> None:
    small_attributes = tuple(f"a{i}" for i in range(9))
    rules = tuple(
        AttributeImplication(
            premise=tuple(bit for bit in range(9) if mask & (1 << bit)),
            conclusion=(),
        )
        for mask in range(MAX_IMPLICATIONS)
    )
    FiniteAttributeImplicationSystem(
        attributes=small_attributes,
        implications=rules,
    )
    with pytest.raises(ValidationError):
        FiniteAttributeImplicationSystem(
            attributes=small_attributes,
            implications=(
                *rules,
                AttributeImplication(premise=(8,), conclusion=(0,)),
            ),
        )


def test_aggregate_membership_and_result_size_boundaries() -> None:
    attributes = tuple(f"{index:02d}" + "x" * 62 for index in range(MAX_ATTRIBUTES))
    premises = tuple(islice(combinations(range(MAX_ATTRIBUTES), 16), MAX_IMPLICATIONS))
    rules = tuple(
        AttributeImplication(premise=premise, conclusion=()) for premise in premises
    )
    assert sum(len(rule.premise) for rule in rules) == MAX_IMPLICATION_MEMBERSHIPS
    system = FiniteAttributeImplicationSystem(
        attributes=attributes,
        implications=rules,
    )
    result = implication_closure(system, frozenset())
    assert (
        len(encode_strict_json(result.model_dump(mode="json")))
        <= MAX_IMPLICATION_CLOSURE_RESULT_BYTES
    )

    extra = next(
        attribute
        for attribute in range(MAX_ATTRIBUTES)
        if attribute not in rules[-1].premise
    )
    over_limit = (
        *rules[:-1],
        AttributeImplication(premise=rules[-1].premise, conclusion=(extra,)),
    )
    with pytest.raises(ValidationError):
        FiniteAttributeImplicationSystem(
            attributes=attributes,
            implications=over_limit,
        )


def test_catalog_adapter_returns_source_bound_result() -> None:
    request = ImplicationClosureRequest(system=_chain_system(), seed=(0,))

    result = compute_implication_closure(request)

    assert isinstance(result, ImplicationClosureResult)
    assert result.system == request.system
    assert result.seed == request.seed
