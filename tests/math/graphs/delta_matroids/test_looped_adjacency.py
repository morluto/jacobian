"""Looped graph adjacency matrices define exact binary delta-matroids."""

from __future__ import annotations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.graphs import LoopedSimpleGraph
from jacobian.math.graphs.delta_matroids import looped_adjacency_delta_matroid
from jacobian.math.graphs.delta_matroids._models import (
    LoopedGraphDeltaMatroidRequest,
    LoopedGraphDeltaMatroidResult,
)
from jacobian.math.graphs.delta_matroids._tools import TOOLS


def _nonsingular_over_gf2(matrix: tuple[tuple[int, ...], ...]) -> bool:
    """Independent row-reduction oracle, including the empty determinant one."""

    rows = [list(row) for row in matrix]
    rank = 0
    for column in range(len(rows)):
        pivot = next((r for r in range(rank, len(rows)) if rows[r][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(rank + 1, len(rows)):
            if rows[r][column]:
                rows[r] = [x ^ y for x, y in zip(rows[r], rows[rank], strict=True)]
        rank += 1
    return rank == len(rows)


def test_looped_graph_maps_diagonal_and_edges_to_matrix_and_feasible_family() -> None:
    graph = LoopedSimpleGraph(
        vertices=("b", "a", "c"), edges=(("a", "b"), ("a", "c")), loops=("a",)
    )
    result = looped_adjacency_delta_matroid(graph)

    assert result.matrix.ground == graph.vertices
    assert result.matrix.entries == ((0, 1, 0), (1, 1, 1), (0, 1, 0))
    expected = []
    for mask in range(1 << len(graph.vertices)):
        subset = tuple(i for i in range(len(graph.vertices)) if mask >> i & 1)
        principal = tuple(
            tuple(result.matrix.entries[i][j] for j in subset) for i in subset
        )
        if _nonsingular_over_gf2(principal):
            expected.append(subset)
    assert result.delta_matroid.feasible == tuple(sorted(expected))
    assert result.graph == graph


def test_empty_graph_preserves_empty_principal_minor_convention() -> None:
    result = looped_adjacency_delta_matroid(
        LoopedSimpleGraph(vertices=(), edges=(), loops=())
    )
    assert result.matrix.entries == ()
    assert result.delta_matroid.ground == ()
    assert result.delta_matroid.feasible == ((),)


def test_wire_result_roundtrip_binds_graph_axis_and_matrix() -> None:
    result = looped_adjacency_delta_matroid(
        LoopedSimpleGraph(vertices=("v",), edges=(), loops=("v",))
    )
    assert (
        LoopedGraphDeltaMatroidResult.model_validate_json(result.model_dump_json())
        == result
    )

    forged = result.model_dump(mode="json")
    forged["matrix"]["entries"] = [[0]]
    with pytest.raises(ValueError, match="adjacency matrix"):
        LoopedGraphDeltaMatroidResult.model_validate(forged)


def test_looped_graph_rejects_duplicate_or_undeclared_loops() -> None:
    with pytest.raises(ValueError):
        LoopedSimpleGraph(vertices=("v",), edges=(), loops=("v", "v"))
    with pytest.raises(ValueError):
        LoopedSimpleGraph(vertices=("v",), edges=(), loops=("other",))


def test_catalog_operation_has_a_real_looped_graph_example() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "graph.looped_adjacency_delta_matroid.compute"
    )
    request = tool.request_type.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert result.delta_matroid.feasible == ((), (0,), (0, 1))


def test_global_catalog_discovers_and_executes_graph_conversion() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("graph.looped_adjacency_delta_matroid.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output["matrix"]["entries"] == [[1, 1], [1, 0]]
    assert result.output["delta_matroid"]["feasible"] == [[], [0], [0, 1]]


def test_native_operation_normalizes_mapping_and_types_admission_errors() -> None:
    graph = {"vertices": [f"v{i}" for i in range(9)], "edges": [], "loops": []}
    with pytest.raises(OperationDomainValidationError):
        looped_adjacency_delta_matroid(graph)  # type: ignore[arg-type]
    canonical = LoopedSimpleGraph.model_construct(
        vertices=tuple(f"v{i}" for i in range(9)), edges=(), loops=()
    )
    with pytest.raises(OperationResourceAdmissionError):
        looped_adjacency_delta_matroid(canonical)


def test_operation_bounds_graph_before_adjacency_matrix_expansion() -> None:
    graph = LoopedSimpleGraph(
        vertices=tuple(f"v{i}" for i in range(9)), edges=(), loops=()
    )
    with pytest.raises(OperationResourceAdmissionError, match="at most 8 vertices"):
        looped_adjacency_delta_matroid(graph)
    assert len(LoopedGraphDeltaMatroidRequest.model_validate({"graph": graph.model_dump()}).graph.vertices) == 9
