"""Dispatch boundary for the canonical graph-homomorphism check."""

from typing import Any

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation


def _canonical_payload() -> dict[str, Any]:
    return {
        "vertex_map": {
            "source_graph": {
                "vertices": ["a", "b"],
                "edges": [["a", "b"]],
            },
            "target_graph": {
                "vertices": ["x", "y"],
                "edges": [["x", "y"]],
            },
            "rows": [
                {"source_vertex": "a", "target_vertex": "x"},
                {"source_vertex": "b", "target_vertex": "y"},
            ],
        }
    }


def test_dispatch_runs_canonical_graph_homomorphism_check() -> None:
    result = invoke_operation(
        "graph.homomorphism.check",
        _canonical_payload(),
        Catalog.open(),
    )

    assert result.output["status"] == "HOMOMORPHISM"
    vertex_map = result.output["homomorphism"]["vertex_map"]
    assert vertex_map["rows"] == _canonical_payload()["vertex_map"]["rows"]


def test_dispatch_admits_a_near_limit_positive_map_without_duplicate_storage() -> None:
    label_width = 18_000
    source_vertices = [
        f"{index:03d}-" + "s" * (label_width - 4) for index in range(256)
    ]
    payload = {
        "vertex_map": {
            "source_graph": {"vertices": source_vertices, "edges": []},
            "target_graph": {"vertices": ["t"], "edges": []},
            "rows": [
                {"source_vertex": source_vertex, "target_vertex": "t"}
                for source_vertex in source_vertices
            ],
        }
    }

    result = invoke_operation("graph.homomorphism.check", payload, Catalog.open())

    assert result.output["status"] == "HOMOMORPHISM"
    assert (
        result.output["homomorphism"]["vertex_map"]["rows"]
        == payload["vertex_map"]["rows"]
    )


def test_dispatch_rejects_a_noncanonical_graph_payload() -> None:
    with pytest.raises(OperationRequestValidationError):
        invoke_operation(
            "graph.homomorphism.check",
            {
                "source_graph": {"vertex_count": 2, "edges": [[0, 1]]},
                "target_graph": {"vertex_count": 2, "edges": [[0, 1]]},
                "vertex_map": [0, 1],
            },
            Catalog.open(),
        )
