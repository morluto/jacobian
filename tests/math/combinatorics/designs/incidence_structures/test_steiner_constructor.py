"""Bounded exact Steiner triple-system construction tests (#1666)."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.combinatorics.designs.incidence_structures._models import (
    IncidenceStructure,
    SteinerTripleSystemRequest,
    SteinerTripleSystemResult,
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


def test_budget_exhaustion_is_unknown() -> None:
    result = _steiner_triple_system(
        SteinerTripleSystemRequest(order=7, search_budget=1)
    )
    assert result.status == "UNKNOWN"
    assert result.design is None


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


def test_computed_result_rejects_non_steiner_design() -> None:
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
    with pytest.raises(ValidationError, match="exactly one block"):
        SteinerTripleSystemResult(
            status="COMPUTED", order=7, design=design, states_explored=1
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


def test_constructor_executes_through_public_catalog_boundary() -> None:
    result = invoke_operation(
        "combinatorics.design.steiner_triple_system.construct",
        {"order": 7, "search_budget": 100_000},
        Catalog.open(),
    )
    assert result.output["status"] == "COMPUTED"
    assert len(result.output["design"]["blocks"]) == 7
