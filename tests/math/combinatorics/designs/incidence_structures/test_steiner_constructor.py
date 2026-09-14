"""Bounded exact Steiner triple-system construction tests (#1666)."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterator
from itertools import combinations
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.designs.incidence_structures import _models as models
from jacobian.math.combinatorics.designs.incidence_structures._models import (
    MAX_STEINER_BLOCKS,
    ComputedSteinerTripleSystem,
    IncidenceStructure,
    SteinerTripleSystemNotFound,
    SteinerTripleSystemRequest,
    SteinerTripleSystemResult,
    SteinerTripleSystemShard,
    SteinerTripleSystemUnknown,
)
from jacobian.math.combinatorics.designs.incidence_structures._tools import (
    _steiner_triple_system,
)
from jacobian.math.combinatorics.designs.incidence_structures.operations import (
    construct_steiner_triple_system,
    incidence_matrix,
)


def test_construct_fano_plane_and_replay_pairs() -> None:
    result = _steiner_triple_system(
        SteinerTripleSystemRequest(order=7, search_budget=100_000)
    )
    assert result.outcome.status == "COMPUTED"
    assert result.outcome.design is not None
    assert len(result.outcome.design.blocks) == 7
    pairs: list[tuple[str, str]] = []
    for block in result.outcome.design.blocks:
        assert len(block) == 3
        pairs.extend(combinations(block, 2))
    assert len(pairs) == 21
    assert Counter(pairs) == Counter(combinations(result.outcome.design.points, 2))


@pytest.mark.parametrize("order", (3, 7, 9, 13, 15))
def test_default_budget_constructs_every_admitted_order(order: int) -> None:
    result = construct_steiner_triple_system(order, 100_000)
    assert result.outcome.status == "COMPUTED"
    assert result.outcome.design is not None
    pairs = Counter(
        pair
        for block in result.outcome.design.blocks
        for pair in combinations(block, 2)
    )
    assert pairs == Counter(combinations(result.outcome.design.points, 2))
    point_index = {
        point: index for index, point in enumerate(result.outcome.design.points)
    }
    assert tuple(
        tuple(point_index[point] for point in block)
        for block in result.outcome.design.blocks
    ) == tuple(
        sorted(
            tuple(point_index[point] for point in block)
            for block in result.outcome.design.blocks
        )
    )


def test_construct_trivial_sts3() -> None:
    result = _steiner_triple_system(
        SteinerTripleSystemRequest(order=3, search_budget=100)
    )
    assert result.outcome.status == "COMPUTED"
    assert result.outcome.design is not None
    assert result.outcome.design.blocks == (("p0", "p1", "p2"),)


def test_native_constructor_uses_request_default_budget() -> None:
    result = construct_steiner_triple_system(3)
    assert result.outcome.status == "COMPUTED"
    assert result.outcome.design.blocks == (("p0", "p1", "p2"),)


def test_budget_exhaustion_is_unknown() -> None:
    result = _steiner_triple_system(
        SteinerTripleSystemRequest(order=7, search_budget=1)
    )
    assert result.outcome.status == "UNKNOWN"
    assert isinstance(result.outcome, SteinerTripleSystemUnknown)
    assert result.outcome.unresolved_frontier


def test_unknown_frontier_can_resume_exact_cover_search() -> None:
    limited = construct_steiner_triple_system(7, 1)
    assert limited.outcome.status == "UNKNOWN"
    assert limited.outcome.unresolved_frontier
    resumed = construct_steiner_triple_system(
        7, 100_000, limited.outcome.unresolved_frontier[0]
    )
    assert resumed.outcome.status == "COMPUTED"
    assert resumed.outcome.design is not None


def test_shard_requires_canonical_in_range_triples() -> None:
    with pytest.raises(ValidationError, match="sorted, distinct, and in range"):
        SteinerTripleSystemShard(order=7, fixed_triples=((0, 2, 1),))
    with pytest.raises(ValidationError, match="must be unique"):
        SteinerTripleSystemShard(order=7, fixed_triples=((0, 1, 2), (0, 1, 2)))


def test_non_array_fixed_triples_are_validation_errors() -> None:
    with pytest.raises(ValidationError):
        SteinerTripleSystemShard.model_validate({"order": 7, "fixed_triples": {}})
    with pytest.raises(ValidationError):
        SteinerTripleSystemShard.model_validate({"order": 7, "fixed_triples": ""})


def test_fixed_triple_family_is_lexicographically_canonical() -> None:
    """An unordered block family has one serialization, independent of tuple order."""
    first = SteinerTripleSystemShard(order=7, fixed_triples=((0, 3, 6), (0, 1, 2)))
    second = SteinerTripleSystemShard(order=7, fixed_triples=((0, 1, 2), (0, 3, 6)))
    assert first == second
    assert first.fixed_triples == ((0, 1, 2), (0, 3, 6))
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_native_continuation_rejects_mismatched_shard_order() -> None:
    with pytest.raises(OperationDomainValidationError, match="same order"):
        construct_steiner_triple_system(
            7, 100, SteinerTripleSystemShard(order=9, fixed_triples=())
        )


def test_native_continuation_revalidates_a_forged_shard() -> None:
    forged = SteinerTripleSystemShard(order=7, fixed_triples=()).model_copy(
        update={"fixed_triples": ((0, 99, 100),)}
    )
    with pytest.raises(OperationDomainValidationError, match="in range"):
        construct_steiner_triple_system(7, 100, forged)


def test_necessary_parameter_condition_rejects_order() -> None:
    with pytest.raises(ValidationError, match="congruent to 1 or 3"):
        SteinerTripleSystemRequest(order=5, search_budget=100)


def test_native_admission_rejects_invalid_order_before_materialization() -> None:
    with pytest.raises(
        OperationDomainValidationError, match="congruent to 1 or 3"
    ) as exc_info:
        construct_steiner_triple_system(5, 100)
    assert exc_info.value.errors()[0]["type"] == (
        "incidence_structure.steiner_order_necessary_condition"
    )


def test_native_admission_uses_semantic_result_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "MAX_STEINER_RESULT_ALLOCATION_UNITS", 1)
    with pytest.raises(
        OperationDomainValidationError, match="retained result allocation"
    ):
        construct_steiner_triple_system(7, 100)


def test_computed_result_rejects_noncanonical_design_axes() -> None:
    design = IncidenceStructure(
        points=("p0", "p1", "p2", "p3", "p4", "p5", "p6"),
        block_ids=tuple(f"b{index}" for index in range(7)),
        blocks=(
            ("p0", "p1", "p2"),
            ("p0", "p1", "p3"),
            ("p0", "p1", "p4"),
            ("p0", "p2", "p3"),
            ("p0", "p2", "p4"),
            ("p0", "p3", "p4"),
            ("p0", "p5", "p6"),
        ),
    )
    with pytest.raises(ValidationError, match="canonical block IDs"):
        SteinerTripleSystemResult(
            order=7,
            outcome=ComputedSteinerTripleSystem(
                design=design.model_copy(
                    update={
                        "block_ids": tuple(f"x{index}" for index in range(7)),
                    }
                ),
            ),
        )


def test_result_round_trip_preserves_composable_design() -> None:
    result = construct_steiner_triple_system(7, 100_000)
    decoded = SteinerTripleSystemResult.model_validate_json(result.model_dump_json())
    assert decoded == result
    # The discriminated branch is what makes this narrowing available: the
    # decoder learns `design` exists from `status`, not from a runtime check.
    assert isinstance(decoded.outcome, ComputedSteinerTripleSystem)
    assert isinstance(result.outcome, ComputedSteinerTripleSystem)
    assert decoded.outcome.design == result.outcome.design
    matrix = incidence_matrix(decoded.outcome.design)
    assert matrix.points == decoded.outcome.design.points
    assert matrix.block_ids == decoded.outcome.design.block_ids
    assert matrix.matrix.row_count == 7
    assert matrix.matrix.column_count == 7


def test_package_exports_the_native_constructor() -> None:
    from jacobian.math.combinatorics.designs import incidence_structures

    assert incidence_structures.construct_steiner_triple_system is (
        construct_steiner_triple_system
    )
    assert "construct_steiner_triple_system" in incidence_structures.__all__


def test_infeasible_continuation_retains_the_source_shard() -> None:
    """NO_COVER on a continuation is shard-local, not global nonexistence."""
    shard = SteinerTripleSystemShard(
        order=13,
        fixed_triples=(
            (0, 1, 2),
            (0, 3, 4),
            (0, 5, 6),
            (0, 7, 8),
            (0, 9, 10),
            (0, 11, 12),
            (1, 3, 5),
            (1, 4, 7),
            (1, 6, 9),
            (1, 8, 11),
            (1, 10, 12),
            (2, 3, 8),
            (3, 7, 9),
            (3, 6, 12),
            (3, 10, 11),
            (6, 8, 10),
            (4, 6, 11),
            (2, 6, 7),
            (5, 7, 10),
        ),
    )
    result = construct_steiner_triple_system(13, 100_000, shard)
    assert result.outcome.status == "NOT_FOUND"
    assert isinstance(result.outcome, SteinerTripleSystemNotFound)
    assert result.outcome.source_shard == shard
    complete = construct_steiner_triple_system(13, 100_000)
    assert complete.outcome.status == "COMPUTED"


def test_computed_result_omits_request_provenance() -> None:
    """An empty-shard and an unsharded COMPUTED result share one encoding."""
    unsharded = construct_steiner_triple_system(7, 100_000)
    sharded = construct_steiner_triple_system(
        7, 100_000, SteinerTripleSystemShard(order=7, fixed_triples=())
    )
    assert unsharded.outcome.status == "COMPUTED"
    assert sharded.outcome.status == "COMPUTED"
    # A COMPUTED branch carries neither a frontier nor shard provenance, so the
    # two routes have one canonical encoding by construction.
    assert isinstance(unsharded.outcome, ComputedSteinerTripleSystem)
    assert isinstance(sharded.outcome, ComputedSteinerTripleSystem)
    assert unsharded.model_dump_json() == sharded.model_dump_json()


def test_computed_branch_has_no_place_for_shard_provenance() -> None:
    """The COMPUTED branch cannot carry shard provenance, by construction.

    Previously one flat model published `source_shard` as independently
    optional and an after-validator rejected the contradiction at runtime, so a
    schema-driven caller could not see that a COMPUTED result never retains it.
    The discriminated branch makes that a shape error instead.
    """

    result = construct_steiner_triple_system(7, 100_000)
    payload = result.model_dump()
    assert "source_shard" not in payload["outcome"]
    payload["outcome"]["source_shard"] = SteinerTripleSystemShard(
        order=7, fixed_triples=()
    ).model_dump()
    with pytest.raises(ValidationError):
        SteinerTripleSystemResult.model_validate(payload)
    with pytest.raises(ValidationError):
        ComputedSteinerTripleSystem.model_validate(
            {
                "status": "COMPUTED",
                "states_explored": 1,
                "design": result.outcome.model_dump()["design"],
                "unresolved_frontier": [],
            }
        )


def test_generated_schema_encodes_each_outcome_branch() -> None:
    """A caller can read the status-to-payload guarantee from the schema alone.

    The generated JSON Schema must discriminate on `status` and require the
    branch payload, so `design` is required exactly on COMPUTED and
    `unresolved_frontier` is required and non-empty exactly on UNKNOWN.
    """

    schema = SteinerTripleSystemResult.model_json_schema(mode="serialization")
    outcome = schema["properties"]["outcome"]
    assert outcome["discriminator"]["propertyName"] == "status"
    branches = {
        definition["properties"]["status"]["const"]: definition
        for definition in (
            schema["$defs"][ref.rsplit("/", 1)[-1]]
            for ref in outcome["discriminator"]["mapping"].values()
        )
    }
    assert set(branches) == {"COMPUTED", "NOT_FOUND", "UNKNOWN"}
    assert "design" in branches["COMPUTED"]["required"]
    assert "source_shard" not in branches["COMPUTED"]["properties"]
    assert "unresolved_frontier" not in branches["COMPUTED"]["properties"]
    assert "design" not in branches["UNKNOWN"]["properties"]
    assert "unresolved_frontier" in branches["UNKNOWN"]["required"]
    assert branches["UNKNOWN"]["properties"]["unresolved_frontier"]["minItems"] == 1
    assert "design" not in branches["NOT_FOUND"]["properties"]
    assert "unresolved_frontier" not in branches["NOT_FOUND"]["properties"]


def test_nonsemantic_shard_prefix_is_a_typed_domain_error() -> None:
    """Overlapping pair constraints are rejected independently of traversal."""
    with pytest.raises(OperationDomainValidationError, match="distinct pairs"):
        construct_steiner_triple_system(
            7,
            100,
            SteinerTripleSystemShard(order=7, fixed_triples=((0, 1, 2), (0, 1, 3))),
        )


def test_continuation_treats_fixed_triples_as_block_constraints() -> None:
    """A first-item non-choice is still a valid included Steiner block."""
    result = construct_steiner_triple_system(
        7, 100_000, SteinerTripleSystemShard(order=7, fixed_triples=((0, 2, 3),))
    )
    assert result.outcome.status == "COMPUTED"
    assert result.outcome.design is not None
    assert ("p0", "p2", "p3") in result.outcome.design.blocks


def test_native_constructor_raises_typed_domain_errors() -> None:
    """Wrong native argument types use the declared domain error, not TypeError."""
    for order, budget in ((True, 100), (7, 1.0)):
        with pytest.raises(OperationDomainValidationError) as error:
            construct_steiner_triple_system(order, budget)  # type: ignore[arg-type]
        assert error.value.errors()[0]["type"] == (
            "incidence_structure.steiner_argument_type"
        )
    with pytest.raises(OperationDomainValidationError) as shard_error:
        construct_steiner_triple_system(7, 100, "not-a-shard")  # type: ignore[arg-type]
    assert shard_error.value.errors()[0]["type"] == (
        "incidence_structure.steiner_shard_type"
    )


def test_forged_shard_is_bounded_before_canonicalization() -> None:
    """An oversized forged prefix is refused without traversing every triple."""
    forged = SteinerTripleSystemShard.model_construct(
        order=7, fixed_triples=tuple((0, 1, 2) for _ in range(500_000))
    )
    started = time.monotonic()
    with pytest.raises(OperationDomainValidationError) as error:
        construct_steiner_triple_system(7, 100, forged)
    assert error.value.errors()[0]["type"] == (
        "incidence_structure.steiner_shard_length"
    )
    assert time.monotonic() - started < 1.0


def test_oversized_forged_shard_is_measured_before_its_elements_are_inspected() -> None:
    """Prove the length bound runs first, without relying on wall time.

    The previous order evaluated `any(not isinstance(...))` inside the same
    condition as the type test, so a schema-bypassed prefix was traversed in
    full before `len` was consulted - unbounded CPU, because this admission runs
    before the request deadline is bound and so has no checkpoint to notice.
    This field reports an oversized length and fails loudly if anything walks
    it, which pins the order deterministically.
    """

    class _ForgedShardField(tuple):  # type: ignore[type-arg]
        declared_length: int

        def __new__(cls, declared_length: int) -> _ForgedShardField:
            forged = tuple.__new__(cls)
            forged.declared_length = declared_length
            return forged

        def __len__(self) -> int:
            return self.declared_length

        def __iter__(self) -> Iterator[Any]:
            raise AssertionError(
                "bounded shard admission must not traverse an oversized prefix"
            )

    forged = SteinerTripleSystemShard.model_construct(
        order=7, fixed_triples=_ForgedShardField(400_000)
    )
    with pytest.raises(OperationDomainValidationError) as error:
        construct_steiner_triple_system(7, 100, forged)
    # A tuple subclass is refused outright, before its length is even read: a
    # subclass can report a length under the ceiling while iteration still
    # yields the whole underlying family.
    assert error.value.errors()[0]["type"] == (
        "incidence_structure.steiner_shard_shape"
    )

    # An exact oversized tuple is still measured before its elements are read.
    exact_oversized = SteinerTripleSystemShard.model_construct(
        order=7, fixed_triples=tuple([0, 1, 2] for _ in range(200_000))
    )
    with pytest.raises(OperationDomainValidationError) as exact_error:
        construct_steiner_triple_system(7, 100, exact_oversized)
    assert exact_error.value.errors()[0]["type"] == (
        "incidence_structure.steiner_shard_length"
    )

    # Malformed *and* oversized still reports the length, which the element scan
    # used to raise first.
    oversized_lists = SteinerTripleSystemShard.model_construct(
        order=7, fixed_triples=tuple([0, 1, 2] for _ in range(200_000))
    )
    with pytest.raises(OperationDomainValidationError) as error:
        construct_steiner_triple_system(7, 100, oversized_lists)
    assert error.value.errors()[0]["type"] == (
        "incidence_structure.steiner_shard_length"
    )


def test_empty_shard_normalizes_to_the_root_encoding() -> None:
    """An empty shard and an unsharded UNKNOWN result serialize identically."""
    sharded = construct_steiner_triple_system(
        7, 1, SteinerTripleSystemShard(order=7, fixed_triples=())
    )
    unsharded = construct_steiner_triple_system(7, 1)
    assert sharded.outcome.status == "UNKNOWN"
    assert sharded.outcome.source_shard is None
    assert sharded.model_dump_json() == unsharded.model_dump_json()


@pytest.mark.parametrize("order", (7, 9, 13))
def test_sharded_completion_replays_pair_coverage(order: int) -> None:
    """The fixed prefix and the completed cover still form one STS.

    Result construction no longer re-derives the defining incidence axiom: the
    cover search returns selected rows only after it has reconstructed and
    checked exact coverage of every remaining pair constraint, and the shard
    admission already established that the fixed prefix is pair-disjoint. This
    is where that composition is checked independently of the kernel.
    """

    # Seed from a first block of the canonical design so the continuation route
    # is exercised rather than the unsharded root search.
    root = construct_steiner_triple_system(order, 100_000)
    assert root.outcome.status == "COMPUTED"
    first_block = root.outcome.design.blocks[0]
    points = tuple(f"p{index}" for index in range(order))
    lowest, middle, highest = sorted(points.index(point) for point in first_block)
    fixed: tuple[int, int, int] = (lowest, middle, highest)

    result = construct_steiner_triple_system(
        order, 100_000, SteinerTripleSystemShard(order=order, fixed_triples=(fixed,))
    )
    assert result.outcome.status == "COMPUTED"
    design = result.outcome.design
    assert fixed in tuple(
        tuple(sorted(points.index(point) for point in block)) for block in design.blocks
    )
    pairs = Counter(pair for block in design.blocks for pair in combinations(block, 2))
    assert pairs == Counter(combinations(design.points, 2))
    assert len(design.blocks) == order * (order - 1) // 6


def test_duplicate_triples_use_a_duplicate_specific_error_code() -> None:
    """A malformed prefix is distinguishable from a request/shard order mismatch.

    `steiner_shard_order` already reports "a continuation shard must have the
    same order as the request", so reusing it for duplicates left callers unable
    to tell a malformed `fixed_triples` field from the actionable need to change
    the requested order.
    """

    with pytest.raises(ValidationError) as error:
        SteinerTripleSystemShard(order=7, fixed_triples=((0, 1, 2), (0, 1, 2)))
    assert error.value.errors()[0]["type"] == (
        "incidence_structure.steiner_shard_duplicate"
    )

    with pytest.raises(OperationDomainValidationError) as mismatch:
        construct_steiner_triple_system(
            7, 100, SteinerTripleSystemShard(order=9, fixed_triples=())
        )
    assert mismatch.value.errors()[0]["type"] == (
        "incidence_structure.steiner_shard_order"
    )


def test_forged_shard_without_order_is_a_typed_domain_error() -> None:
    """A schema-bypassed shard missing ``order`` is not an AttributeError."""

    forged = SteinerTripleSystemShard.model_construct(fixed_triples=())
    with pytest.raises(OperationDomainValidationError) as error:
        construct_steiner_triple_system(7, 100, forged)
    assert error.value.errors()[0]["loc"] == ("shard", "order")


def test_oversized_wire_shard_is_measured_before_it_is_canonicalized() -> None:
    """An over-long wire family is refused on `len`, not copied and sorted.

    The before-validator converted every inner list and sorted the whole family
    ahead of the field's own `max_length`, so a request that is guaranteed to be
    rejected still paid work proportional to an arbitrarily large input.
    """

    family = [(0, 1, 2)] * (MAX_STEINER_BLOCKS + 1)
    with pytest.raises(ValidationError) as error:
        SteinerTripleSystemShard.model_validate({"order": 7, "fixed_triples": family})
    assert error.value.errors()[0]["type"] == "too_long"


def test_wire_shard_rejects_sequence_subclasses_before_canonicalization() -> None:
    """A sequence subclass cannot understate its length and force a traversal.

    ``isinstance`` admitted list/tuple subclasses whose ``__len__`` reported a
    value under the ceiling, so the copy and sort traversed the whole underlying
    family before the field's ``max_length`` could reject it.
    """

    class _LyingFamily(tuple):  # type: ignore[type-arg]
        def __len__(self) -> int:
            return 1

        def __iter__(self) -> Iterator[Any]:
            raise AssertionError("a rejected family must not be traversed")

    family = _LyingFamily(((0, 1, 2),))
    with pytest.raises(ValidationError) as error:
        SteinerTripleSystemShard.model_validate({"order": 7, "fixed_triples": family})
    assert error.value.errors()[0]["type"] == (
        "incidence_structure.steiner_shard_shape"
    )


def test_native_shard_bound_does_not_come_from_the_unadmitted_order() -> None:
    """A forged order cannot widen the prefix ceiling.

    `order * (order - 1) // 6` is derived from a request value that has not been
    range-admitted when the shard is inspected, so a forged large `order` used
    to authorise a family of any size to reach Pydantic canonicalization and
    sorting. The ceiling is now the fixed structural maximum, which no admitted
    design exceeds.
    """

    forged = SteinerTripleSystemShard.model_construct(
        order=10**6,
        fixed_triples=tuple((0, 1, 2) for _ in range(MAX_STEINER_BLOCKS + 5)),
    )
    # The ceiling was `order * (order - 1) // 6` over the *request* order, which
    # is range-admitted only after this inspection, so a forged large order let
    # the whole family through to Pydantic canonicalization and surfaced a raw
    # `less_than_equal` instead of the shard diagnostic.
    with pytest.raises(OperationDomainValidationError) as error:
        construct_steiner_triple_system(10**6, 100, forged)
    assert error.value.errors()[0]["type"] == (
        "incidence_structure.steiner_shard_length"
    )


def test_tuple_subclass_cannot_understate_the_prefix_length() -> None:
    """A lying `__len__` must not authorise an unbounded traversal.

    `isinstance(raw_triples, tuple)` admitted subclasses, so a forged shard
    could report a length under the ceiling while iteration still yielded the
    whole underlying family, letting the element scan and Pydantic
    canonicalization run without bound.
    """

    class _UnderstatingTuple(tuple):  # type: ignore[type-arg]
        def __len__(self) -> int:
            return 1

        def __iter__(self) -> Iterator[Any]:
            raise AssertionError("a subverted length must not reach the scan")

    forged = SteinerTripleSystemShard.model_construct(
        order=7, fixed_triples=_UnderstatingTuple([(0, 1, 2)] * 50_000)
    )
    with pytest.raises(OperationDomainValidationError) as error:
        construct_steiner_triple_system(7, 100, forged)
    assert error.value.errors()[0]["type"] == (
        "incidence_structure.steiner_shard_shape"
    )


def test_unresolved_frontier_has_one_canonical_encoding() -> None:
    """The frontier is an unordered family, so order and duplicates are erased.

    Two payloads carrying the same shards in different orders used to validate
    and serialize differently, exposing private DFS ordering as wire-visible
    structure and admitting several encodings of one continuation state.
    """

    first = SteinerTripleSystemShard(order=7, fixed_triples=((0, 1, 2),))
    second = SteinerTripleSystemShard(order=7, fixed_triples=((0, 1, 3),))
    reversed_payload = SteinerTripleSystemResult(
        order=7,
        outcome=SteinerTripleSystemUnknown(
            states_explored=4, unresolved_frontier=(second, first)
        ),
    )
    forward_payload = SteinerTripleSystemResult(
        order=7,
        outcome=SteinerTripleSystemUnknown(
            states_explored=4, unresolved_frontier=(first, second)
        ),
    )
    assert reversed_payload.model_dump_json() == forward_payload.model_dump_json()

    duplicated = SteinerTripleSystemResult(
        order=7,
        outcome=SteinerTripleSystemUnknown(
            states_explored=4, unresolved_frontier=(first, second, first)
        ),
    )
    assert isinstance(duplicated.outcome, SteinerTripleSystemUnknown)
    assert duplicated.outcome.unresolved_frontier == (first, second)

    # An empty frontier is still refused: the field requires at least one.
    with pytest.raises(ValidationError):
        SteinerTripleSystemUnknown(states_explored=1, unresolved_frontier=())


def test_wire_frontier_shards_are_canonicalized_after_decoding() -> None:
    """Wire shard dicts canonicalize like already-instantiated shards.

    ``model_validate_json`` supplies each frontier shard as a dict, so the
    instance-only guard left reversed or duplicated wire payloads as distinct
    serialized continuation states.
    """

    payload = {
        "order": 7,
        "outcome": {
            "status": "UNKNOWN",
            "states_explored": 4,
            "unresolved_frontier": [
                {"order": 7, "fixed_triples": [[0, 1, 2]]},
                {"order": 7, "fixed_triples": [[0, 1, 3]]},
            ],
        },
    }
    forward = SteinerTripleSystemResult.model_validate_json(json.dumps(payload))

    reversed_payload = json.loads(json.dumps(payload))
    reversed_payload["outcome"]["unresolved_frontier"].reverse()
    reversed_result = SteinerTripleSystemResult.model_validate_json(
        json.dumps(reversed_payload)
    )
    assert reversed_result.model_dump_json() == forward.model_dump_json()

    duplicated_payload = json.loads(json.dumps(payload))
    duplicated_payload["outcome"]["unresolved_frontier"].append(
        {"order": 7, "fixed_triples": [[0, 1, 2]]}
    )
    duplicated = SteinerTripleSystemResult.model_validate_json(
        json.dumps(duplicated_payload)
    )
    assert isinstance(duplicated.outcome, SteinerTripleSystemUnknown)
    assert len(duplicated.outcome.unresolved_frontier) == 2


def test_wire_frontier_shard_rejects_fixed_triple_subclasses() -> None:
    """A shard dict whose fixed_triples is a lying subclass is rejected."""

    class LyingTriples(tuple):  # type: ignore[type-arg]
        def __len__(self) -> int:
            return 1

        def __iter__(self) -> Iterator[Any]:
            raise AssertionError("a rejected family must not be traversed")

    payload = {
        "order": 7,
        "outcome": {
            "status": "UNKNOWN",
            "states_explored": 1,
            "unresolved_frontier": [
                {"order": 7, "fixed_triples": LyingTriples(((0, 1, 2),))}
            ],
        },
    }
    with pytest.raises(ValidationError):
        SteinerTripleSystemResult.model_validate(payload)


def test_wire_shard_bounds_each_triple_before_copying() -> None:
    """An inner array of the wrong size is rejected without converting it."""
    with pytest.raises(ValidationError) as error:
        SteinerTripleSystemShard.model_validate(
            {"order": 7, "fixed_triples": [[0] * 4]}
        )
    assert error.value.errors()[0]["type"] == "incidence_structure.steiner_shard_triple"


def test_wire_frontier_rejects_sequence_subclasses() -> None:
    """A frontier sequence subclass is rejected before it is traversed."""

    class LyingFrontier(tuple):  # type: ignore[type-arg]
        def __len__(self) -> int:
            return 1

        def __iter__(self) -> Iterator[Any]:
            raise AssertionError("a rejected frontier must not be traversed")

    payload = {
        "order": 7,
        "outcome": {
            "status": "UNKNOWN",
            "states_explored": 1,
            "unresolved_frontier": LyingFrontier(
                ({"order": 7, "fixed_triples": [[0, 1, 2]]},)
            ),
        },
    }
    with pytest.raises(ValidationError) as error:
        SteinerTripleSystemResult.model_validate(payload)
    assert error.value.errors()[0]["type"] == (
        "incidence_structure.steiner_frontier_shape"
    )


def test_not_found_outcome_omits_execution_history() -> None:
    """An exact NOT_FOUND outcome does not carry private search counts."""
    assert "states_explored" not in SteinerTripleSystemNotFound.model_fields


def test_frontier_rejects_unhashable_forged_shard_fields() -> None:
    """A constructed shard with an unhashable field is a ValidationError."""
    forged = SteinerTripleSystemShard.model_construct(order=7, fixed_triples=[])
    payload = {
        "order": 7,
        "outcome": {
            "status": "UNKNOWN",
            "states_explored": 1,
            "unresolved_frontier": [forged],
        },
    }
    with pytest.raises(ValidationError):
        SteinerTripleSystemResult.model_validate(payload)
