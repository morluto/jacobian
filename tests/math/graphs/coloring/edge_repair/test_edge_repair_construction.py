"""Trusted repair construction scans once while caller claims remain checked."""

import json
from itertools import combinations
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.coloring import _models, operations
from jacobian.math.graphs.coloring._coloring_process import run_coloring_worker
from jacobian.math.graphs.coloring._models import (
    PrecoloringEdgeRepairResult,
    VertexColoringAssignment,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph


@pytest.mark.parametrize("kind", ["empty", "isolated", "forced", "complete"])
def test_repair_construction_scans_once_and_decoding_replays_the_relation(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = IndexedSimpleUndirectedGraph(
        vertex_count=0 if kind == "empty" else 4,
        edges=tuple(combinations(range(4), 2))
        if kind == "complete"
        else ((0, 1),)
        if kind == "forced"
        else (),
    )
    fixed = () if kind == "empty" else ((0, 0), (1, 0), (2, 1), (3, 2))
    expected = (0,) if graph.edges else ()
    real_scan = _models._monochromatic_edge_indices
    real_worker = run_coloring_worker
    scans = 0
    solves = 0

    def tracked_scan(
        source: IndexedSimpleUndirectedGraph, coloring: tuple[int, ...]
    ) -> tuple[int, ...]:
        nonlocal scans
        scans += 1
        return real_scan(source, coloring)

    def tracked_worker(*args: Any, **kwargs: Any) -> Any:
        nonlocal solves
        solves += 1
        return real_worker(*args, **kwargs)

    monkeypatch.setattr(_models, "_monochromatic_edge_indices", tracked_scan)
    monkeypatch.setattr(operations, "run_coloring_worker", tracked_worker)
    result = operations.precoloring_edge_repair(graph, 3, fixed, 100_000)
    assert type(result) is PrecoloringEdgeRepairResult
    assert type(result.coloring) is VertexColoringAssignment
    assert result.graph is graph
    assert result.coloring.graph is graph
    assert result.coloring.coloring == tuple(color for _, color in fixed)
    assert result.repaired_edge_indices == expected
    assert result.repaired_edge_count == len(expected)
    assert scans == 1
    assert solves == bool(graph.edges)

    decoded = PrecoloringEdgeRepairResult.model_validate_json(result.model_dump_json())
    assert decoded == result
    assert scans == 2
    assert solves == bool(graph.edges)


@pytest.fixture(scope="module")
def repair_result() -> PrecoloringEdgeRepairResult:
    return operations.precoloring_edge_repair(
        IndexedSimpleUndirectedGraph(
            vertex_count=4, edges=tuple(combinations(range(4), 2))
        ),
        3,
        ((0, 0), (1, 0), (2, 1), (3, 2)),
        100_000,
    )


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ("count", "graph.precoloring_repair_count_must_match_indices"),
        ("order", "graph.precoloring_repaired_edges_must_be_sorted"),
        ("index_range", "graph.precoloring_repaired_edges_must_be_source_bound"),
        ("wrong_edges", "graph.precoloring_repaired_edges_must_be_monochromatic"),
        ("duplicate_edges", "graph.precoloring_repaired_edges_must_be_monochromatic"),
        ("source", "graph.precoloring_witness_must_bind_source_and_palette"),
        ("palette", "graph.precoloring_witness_must_bind_source_and_palette"),
        ("bound_source", "graph.precoloring_repaired_edges_must_be_monochromatic"),
        (
            "fixed_duplicate",
            "graph.precoloring_fixed_colors_must_assign_one_color_per_vertex",
        ),
        ("fixed_order", "graph.precoloring_fixed_colors_must_be_strictly_increasing"),
        ("fixed_vertex", "graph.precoloring_fixed_vertex_out_of_range"),
        ("fixed_color", "graph.precoloring_fixed_color_out_of_range"),
        ("fixed_extension", "graph.precoloring_witness_must_extend_fixed_colors"),
        ("coloring_length", "graph.coloring_must_assign_one_color_per_vertex"),
        ("coloring_range", "graph.coloring_values_must_be_in_0_colors_1"),
        ("coloring_bool", "int_type"),
        ("solver_budget", "less_than_equal"),
    ],
)
def test_untrusted_repair_claims_retain_all_context_and_relation_checks(
    repair_result: PrecoloringEdgeRepairResult, change: str, code: str
) -> None:
    payload = repair_result.model_dump(mode="json")
    if change == "count":
        payload["repaired_edge_count"] = 2
    elif change == "order":
        payload.update(repaired_edge_count=2, repaired_edge_indices=[1, 0])
    elif change == "index_range":
        payload["repaired_edge_indices"] = [6]
    elif change == "wrong_edges":
        payload["repaired_edge_indices"] = [1]
    elif change == "duplicate_edges":
        payload.update(repaired_edge_count=2, repaired_edge_indices=[0, 0])
    elif change in {"source", "bound_source"}:
        payload["graph"]["edges"] = payload["graph"]["edges"][1:]
        if change == "bound_source":
            payload["coloring"]["graph"] = payload["graph"]
    elif change == "palette":
        payload["colors"] = 2
    elif change.startswith("fixed_"):
        payload["fixed_colors"] = {
            "fixed_duplicate": [[0, 0], [0, 0]],
            "fixed_order": [[1, 0], [0, 0]],
            "fixed_vertex": [[4, 0]],
            "fixed_color": [[0, 3]],
            "fixed_extension": [[0, 1]],
        }[change]
    elif change == "coloring_length":
        payload["coloring"]["coloring"] = [0, 0, 1]
    elif change == "coloring_range":
        payload["coloring"]["coloring"] = [0, 0, 1, 3]
    elif change == "coloring_bool":
        payload["coloring"]["coloring"] = [False, 0, 1, 2]
    else:
        payload["solver_conflicts"] = 1_000_001
    with pytest.raises(ValidationError) as error:
        PrecoloringEdgeRepairResult.model_validate_json(json.dumps(payload))
    assert error.value.errors()[0]["type"] == code


@pytest.mark.parametrize(
    ("coloring", "fixed_colors", "code"),
    [
        ((0,), ((0, 0),), "graph.coloring_must_assign_one_color_per_vertex"),
        ((0, 2), ((0, 0),), "graph.coloring_values_must_be_in_0_colors_1"),
        ((False, 1), ((0, 0),), "int_type"),
        ((0, 1), ((0, 1),), "graph.precoloring_witness_must_extend_fixed_colors"),
        (
            (0, 1),
            ((0, 0), (0, 0)),
            "graph.precoloring_fixed_colors_must_assign_one_color_per_vertex",
        ),
    ],
)
def test_checked_factory_rejects_bad_assignments_before_edge_scan(
    coloring: tuple[int, ...],
    fixed_colors: tuple[tuple[int, int], ...],
    code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_scan = _models._monochromatic_edge_indices
    scans = 0

    def tracked_scan(
        graph: IndexedSimpleUndirectedGraph, assignment: tuple[int, ...]
    ) -> tuple[int, ...]:
        nonlocal scans
        scans += 1
        return real_scan(graph, assignment)

    monkeypatch.setattr(_models, "_monochromatic_edge_indices", tracked_scan)
    with pytest.raises(ValidationError) as error:
        PrecoloringEdgeRepairResult._from_kernel(
            graph=IndexedSimpleUndirectedGraph(vertex_count=2, edges=((0, 1),)),
            colors=2,
            fixed_colors=fixed_colors,
            solver_conflicts=100_000,
            coloring=coloring,
        )
    assert error.value.errors()[0]["type"] == code
    assert scans == 0
