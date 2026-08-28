from typing import Any

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
