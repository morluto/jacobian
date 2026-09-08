from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import MathTool
from jacobian.math.graphs.optimization._invariants import (
    EXACT_GRAPH_INVARIANT_OPERATIONS,
)


def _operation(operation_id: str) -> MathTool[Any, Any]:
    return next(
        operation
        for operation in EXACT_GRAPH_INVARIANT_OPERATIONS
        if operation.operation_id == operation_id
    )


def _path_graph() -> dict[str, object]:
    return {
        "vertices": ["a", "b", "c"],
        "edges": [["a", "b"], ["b", "c"]],
    }


def test_graph_metric_operations_are_published_and_exact() -> None:
    radius = _operation("graph.invariant.radius.compute")
    diameter = _operation("graph.invariant.diameter.compute")
    eulerian = _operation("graph.invariant.is_eulerian.compute")
    triangle_count = _operation("graph.invariant.triangle_count.compute")

    request = {"graph": _path_graph()}
    assert radius.run(radius.request_type.model_validate(request)).radius == 1
    assert diameter.run(diameter.request_type.model_validate(request)).diameter == 2
    assert (
        eulerian.run(eulerian.request_type.model_validate(request)).is_eulerian is False
    )
    assert (
        triangle_count.run(
            triangle_count.request_type.model_validate(request)
        ).triangle_count
        == 0
    )


def test_invariant_requests_retain_256_vertex_result_envelope() -> None:
    girth = _operation("graph.invariant.girth.compute")
    matching = _operation("graph.invariant.maximum_matching.compute")
    cycle_vertices = tuple(f"{index:03d}" for index in range(256))
    cycle_edges = tuple(
        (cycle_vertices[index], cycle_vertices[(index + 1) % 256])
        if cycle_vertices[index] < cycle_vertices[(index + 1) % 256]
        else (cycle_vertices[(index + 1) % 256], cycle_vertices[index])
        for index in range(256)
    )
    cycle = {
        "vertices": list(cycle_vertices),
        "edges": [list(edge) for edge in cycle_edges],
    }
    girth_result = girth.run(girth.request_type.model_validate({"graph": cycle}))
    assert girth_result.girth == 256
    assert type(girth_result).model_validate(girth_result.model_dump()) == girth_result

    matching_vertices = tuple(f"{index:03d}" for index in range(256))
    matching_edges = tuple(
        (matching_vertices[2 * index], matching_vertices[2 * index + 1])
        for index in range(128)
    )
    matching_graph = {
        "vertices": list(matching_vertices),
        "edges": [list(edge) for edge in matching_edges],
    }
    matching_result = matching.run(
        matching.request_type.model_validate({"graph": matching_graph})
    )
    assert matching_result.maximum_matching_cardinality == 128
    assert (
        type(matching_result).model_validate(matching_result.model_dump())
        == matching_result
    )

    oversized_cycle = {
        "vertices": [f"{index:03d}" for index in range(257)],
        "edges": [
            [f"{index:03d}", f"{(index + 1) % 257:03d}"]
            if f"{index:03d}" < f"{(index + 1) % 257:03d}"
            else [f"{(index + 1) % 257:03d}", f"{index:03d}"]
            for index in range(257)
        ],
    }
    with pytest.raises(ValidationError, match="at most 256 vertices"):
        girth.request_type.model_validate({"graph": oversized_cycle})

    oversized_matching = {
        "vertices": [f"{index:03d}" for index in range(258)],
        "edges": [[f"{2 * index:03d}", f"{2 * index + 1:03d}"] for index in range(129)],
    }
    with pytest.raises(ValidationError, match="at most 256 vertices"):
        matching.request_type.model_validate({"graph": oversized_matching})


def test_disconnected_graph_metrics_report_not_applicable() -> None:
    graph = {"vertices": ["a", "b"], "edges": []}
    for operation_id, field in (
        ("graph.invariant.radius.compute", "radius"),
        ("graph.invariant.diameter.compute", "diameter"),
    ):
        operation = _operation(operation_id)
        result = operation.run(operation.request_type.model_validate({"graph": graph}))
        assert result.status == "NOT_APPLICABLE"
        assert getattr(result, field) is None
