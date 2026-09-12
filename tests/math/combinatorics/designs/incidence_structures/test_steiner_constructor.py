"""Bounded exact Steiner triple-system construction tests (#1666)."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.combinatorics.designs.incidence_structures import _models as models
from jacobian.math.combinatorics.designs.incidence_structures._models import (
    IncidenceStructure,
    SteinerTripleSystemRequest,
    SteinerTripleSystemResult,
    SteinerTripleSystemShard,
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
    assert result.status == "COMPUTED"
    assert result.design is not None
    assert len(result.design.blocks) == 7
    pairs: list[tuple[str, str]] = []
    for block in result.design.blocks:
        assert len(block) == 3
        pairs.extend(combinations(block, 2))
    assert len(pairs) == 21
    assert Counter(pairs) == Counter(combinations(result.design.points, 2))


@pytest.mark.parametrize("order", (3, 7, 9, 13, 15))
def test_default_budget_constructs_every_admitted_order(order: int) -> None:
    result = construct_steiner_triple_system(order, 100_000)
    assert result.status == "COMPUTED"
    assert result.design is not None
    pairs = Counter(
        pair for block in result.design.blocks for pair in combinations(block, 2)
    )
    assert pairs == Counter(combinations(result.design.points, 2))
    point_index = {point: index for index, point in enumerate(result.design.points)}
    assert tuple(
        tuple(point_index[point] for point in block) for block in result.design.blocks
    ) == tuple(
        sorted(
            tuple(point_index[point] for point in block)
            for block in result.design.blocks
        )
    )


def test_construct_trivial_sts3() -> None:
    result = _steiner_triple_system(
        SteinerTripleSystemRequest(order=3, search_budget=100)
    )
    assert result.status == "COMPUTED"
    assert result.design is not None
    assert result.design.blocks == (("p0", "p1", "p2"),)


def test_native_constructor_uses_request_default_budget() -> None:
    result = construct_steiner_triple_system(3)
    assert result.status == "COMPUTED"
    assert result.states_explored < 100_000


def test_budget_exhaustion_is_unknown() -> None:
    result = _steiner_triple_system(
        SteinerTripleSystemRequest(order=7, search_budget=1)
    )
    assert result.status == "UNKNOWN"
    assert result.design is None
    assert result.unresolved_frontier


def test_unknown_frontier_can_resume_exact_cover_search() -> None:
    limited = construct_steiner_triple_system(7, 1)
    assert limited.status == "UNKNOWN"
    assert limited.unresolved_frontier
    resumed = construct_steiner_triple_system(
        7, 100_000, limited.unresolved_frontier[0]
    )
    assert resumed.status == "COMPUTED"
    assert resumed.design is not None


def test_shard_requires_canonical_in_range_triples() -> None:
    with pytest.raises(ValidationError, match="sorted, distinct, and in range"):
        SteinerTripleSystemShard(order=7, fixed_triples=((0, 2, 1),))
    with pytest.raises(ValidationError, match="must be unique"):
        SteinerTripleSystemShard(order=7, fixed_triples=((0, 1, 2), (0, 1, 2)))


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
            status="COMPUTED",
            order=7,
            design=design.model_copy(
                update={
                    "block_ids": tuple(f"x{index}" for index in range(7)),
                }
            ),
            states_explored=1,
        )


def test_result_round_trip_preserves_composable_design() -> None:
    result = construct_steiner_triple_system(7, 100_000)
    decoded = SteinerTripleSystemResult.model_validate_json(result.model_dump_json())
    assert decoded == result
    assert decoded.design == result.design
    assert decoded.design is not None
    matrix = incidence_matrix(decoded.design)
    assert matrix.points == decoded.design.points
    assert matrix.block_ids == decoded.design.block_ids
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
    assert result.status == "NOT_FOUND"
    assert result.design is None
    assert result.source_shard == shard
    complete = construct_steiner_triple_system(13, 100_000)
    assert complete.status == "COMPUTED"


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
    assert result.status == "COMPUTED"
    assert result.design is not None
    assert ("p0", "p2", "p3") in result.design.blocks


def test_constructor_executes_through_public_catalog_boundary() -> None:
    result = invoke_operation(
        "combinatorics.design.steiner_triple_system.construct",
        {"order": 7, "search_budget": 100_000},
        Catalog.open(),
    )
    assert result.output["status"] == "COMPUTED"
    assert len(result.output["design"]["blocks"]) == 7
