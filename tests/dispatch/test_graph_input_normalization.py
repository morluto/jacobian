"""Opted-in graph requests preserve exact meaning and authoritative axes."""

import copy
import json
from itertools import product
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.graphs.coloring._models import EdgeColoringAssignment
from jacobian.math.graphs.morphisms._models import GraphVertexMap
from jacobian.math.graphs.values import MAX_SIMPLE_GRAPH_VERTICES

_RAW_GRAPH: dict[str, Any] = {
    "vertices": ["2", "10", "11"],
    "edges": [["2", "10"], ["11", "10"]],
}
_CANONICAL_GRAPH: dict[str, Any] = {
    "vertices": ["2", "10", "11"],
    "edges": [["10", "2"], ["10", "11"]],
}


def _run(operation: str, **arguments: Any) -> dict[str, Any]:
    return invoke_operation(
        operation, {"graph": _RAW_GRAPH, **arguments}, Catalog.open()
    ).output


@pytest.mark.parametrize(
    ("operation", "field", "expected", "arguments"),
    (
        ("graph.invariant.triangle_count.compute", "triangle_count", 0, {}),
        ("graph.invariant.radius.compute", "radius", 1, {}),
        ("graph.invariant.diameter.compute", "diameter", 2, {}),
        ("graph.invariant.is_eulerian.compute", "is_eulerian", False, {}),
        ("graph.invariant.girth.compute", "girth", 0, {}),
        ("graph.invariant.edge_connectivity.compute", "edge_connectivity", 1, {}),
        ("graph.invariant.vertex_connectivity.compute", "vertex_connectivity", 1, {}),
        ("graph.invariant.spanning_tree_count.compute", "spanning_tree_count", 1, {}),
        (
            "graph.invariant.maximum_matching.compute",
            "maximum_matching_cardinality",
            1,
            {},
        ),
        ("graph.k_core.compute", "vertices", ["10", "11", "2"], {"k": 1}),
        ("graph.invariant.chromatic_number.compute", "chromatic_number", 2, {}),
    ),
)
def test_path_invariants_accept_equivalent_endpoint_spelling(
    operation: str, field: str, expected: Any, arguments: dict[str, Any]
) -> None:
    # P3 is a tree, has radius one, two colors, and one matching edge.
    output = _run(operation, **arguments)
    assert output[field] == expected
    if "graph" in output:
        assert output["graph"] == _CANONICAL_GRAPH
    canonical = invoke_operation(
        operation, {"graph": _CANONICAL_GRAPH, **arguments}, Catalog.open()
    ).output
    assert output == canonical


def test_maximum_cut_matches_exhaustive_bipartitions_and_strict_result_decode() -> None:
    output = _run("graph.cut.maximum.compute")
    vertices = _RAW_GRAPH["vertices"]
    brute_force = max(
        sum(
            sides[vertices.index(left)] != sides[vertices.index(right)]
            for left, right in _RAW_GRAPH["edges"]
        )
        for sides in product((0, 1), repeat=len(vertices))
    )
    assert output["cut_value"] == brute_force == 2
    assert output["graph"] == _CANONICAL_GRAPH
    assert output["crossing_edges"] == _CANONICAL_GRAPH["edges"]
    operation = Catalog.open().operation("graph.cut.maximum.compute")
    assert operation is not None
    assert (
        operation.result_type.model_validate_json(
            json.dumps(output), strict=True
        ).model_dump(mode="json")
        == output
    )
    corrupted = copy.deepcopy(output)
    corrupted["graph"]["edges"][0].reverse()
    with pytest.raises(ValidationError) as error:
        operation.result_type.model_validate_json(json.dumps(corrupted), strict=True)
    assert error.value.errors()[0]["loc"] == ("graph",)
    assert (
        error.value.errors()[0]["type"]
        == "graph.edges_must_contain_two_declared_vertices_in_orde"
    )


def test_signed_embedding_result_darts_are_not_normalized_as_undirected_edges() -> None:
    operation = Catalog.open().operation("graph.embedding.nonorientable.check")
    assert operation is not None
    output = invoke_operation(
        operation.operation_id,
        {
            "graph": {
                "vertices": ["a", "b", "c"],
                "edges": [["b", "a"], ["c", "b"], ["c", "a"]],
            },
            "rotations": [[0, 2], [0, 1], [1, 2]],
            "signs": [0, 1, 1],
        },
        Catalog.open(),
    ).output
    tail, head, reverse = output["darts"][0]
    output["darts"][0] = [head, tail, reverse]
    with pytest.raises(ValidationError) as error:
        operation.result_type.model_validate_json(json.dumps(output), strict=True)
    assert (
        error.value.errors()[0]["type"]
        == "combinatorial_map.signed_embedding_dart_source"
    )


def test_distance_matrix_keeps_source_axis_and_its_declared_sorted_row_axis() -> None:
    output = _run("graph.distance_matrix.compute")
    assert output["graph"] == _CANONICAL_GRAPH
    assert output["rows"] == [
        {"source": "10", "distances": [0, 1, 1]},
        {"source": "11", "distances": [1, 0, 2]},
        {"source": "2", "distances": [1, 2, 0]},
    ]


def test_chromatic_certificate_keeps_vertex_aligned_colors_and_weights() -> None:
    colors = [0, 1, 0]
    weights = [
        {"num": "1", "den": "3"},
        {"num": "1", "den": "1"},
        {"num": "2", "den": "3"},
    ]
    output = _run(
        "graph.coloring.chromatic_number.check",
        claimed_chromatic_number=2,
        coloring=colors,
        weights=weights,
    )
    # The center is one color class; the leaves form the other. Each has weight one.
    assert output["verdict"] == "ACCEPTED"
    assert output["graph"] == _CANONICAL_GRAPH
    assert output["coloring"] == colors
    assert output["weights"] == weights
    assert output["weight_sum"] == {"num": "2", "den": "1"}
    assert output["certified_lower_bound"] == 2


def test_edge_coloring_output_composes_unchanged_with_strict_assignment_consumer() -> (
    None
):
    output = _run("graph.edge_coloring.k_decide", colors=2)
    assert output["colorable"] is True
    assignment = output["coloring"]
    assert assignment["graph"] == _CANONICAL_GRAPH
    assert assignment["coloring"][0] != assignment["coloring"][1]
    checked = invoke_operation(
        "graph.edge_coloring.check", {"assignment": assignment}, Catalog.open()
    ).output
    assert checked["proper"] is True
    assert checked["assignment"] == assignment
    noncanonical = copy.deepcopy(assignment)
    noncanonical["graph"]["edges"][0].reverse()
    assert (
        invoke_operation(
            "graph.edge_coloring.check", {"assignment": noncanonical}, Catalog.open()
        ).output
        == checked
    )
    with pytest.raises(ValidationError) as error:
        EdgeColoringAssignment.model_validate_json(
            json.dumps(noncanonical), strict=True
        )
    assert error.value.errors()[0]["loc"] == ("graph",)


def test_list_capacity_colors_follow_edge_axis_not_endpoint_or_list_sort_order() -> (
    None
):
    output = _run(
        "graph.edge_coloring.list_capacity.assign",
        palette=["red", "blue"],
        lists=[
            {"edge": ["10", "11"], "colors": ["blue"]},
            {"edge": ["10", "2"], "colors": ["red"]},
        ],
        capacities=[{"color": "red", "capacity": 1}, {"color": "blue", "capacity": 1}],
    )
    assert output["status"] == "FEASIBLE"
    assert output["graph"] == _CANONICAL_GRAPH
    assert output["assignment"] == ["red", "blue"]
    assert [entry["edge"] for entry in output["lists"]] == [["10", "11"], ["10", "2"]]


@pytest.mark.parametrize(
    "graph", ({"vertices": [], "edges": []}, {"vertices": ["z"], "edges": []})
)
def test_empty_and_singleton_graphs_keep_their_degenerate_values(
    graph: dict[str, Any],
) -> None:
    output = invoke_operation(
        "graph.cut.maximum.compute", {"graph": graph}, Catalog.open()
    ).output
    assert output["graph"] == graph
    assert output["cut_value"] == 0
    assert output["crossing_edges"] == []


def test_dispatch_does_not_mutate_callers_payload_or_sort_lists() -> None:
    payload = {"graph": copy.deepcopy(_RAW_GRAPH)}
    before = copy.deepcopy(payload)
    invoke_operation("graph.distance_matrix.compute", payload, Catalog.open())
    assert payload == before


@pytest.mark.parametrize(
    ("edges", "code", "location"),
    (
        ([["2", "10"], ["10", "2"]], "graph.graph_edges_must_be_unique", ("graph",)),
        (
            [["2", "2"]],
            "graph.edges_must_contain_two_declared_vertices_in_orde",
            ("graph",),
        ),
        (
            [["missing", "2"]],
            "graph.edges_must_contain_two_declared_vertices_in_orde",
            ("graph",),
        ),
        ([[False, "2"]], "string_type", ("graph", "edges", 0, 0)),
        ([["2"]], "missing", ("graph", "edges", 0, 1)),
    ),
)
def test_dispatch_preserves_structured_input_failures(
    edges: list[list[Any]], code: str, location: tuple[str | int, ...]
) -> None:
    with pytest.raises(OperationRequestValidationError) as error:
        invoke_operation(
            "graph.cut.maximum.compute",
            {"graph": {"vertices": _RAW_GRAPH["vertices"], "edges": edges}},
            Catalog.open(),
        )
    assert error.value.errors()[0]["type"] == code
    assert error.value.errors()[0]["loc"] == location


def test_normalization_does_not_widen_operation_vertex_admission() -> None:
    payload = {
        "graph": {
            "vertices": [str(index) for index in range(MAX_SIMPLE_GRAPH_VERTICES + 1)],
            "edges": [["2", "10"]],
        }
    }
    with pytest.raises(OperationRequestValidationError) as error:
        invoke_operation("graph.cut.maximum.compute", payload, Catalog.open())
    assert error.value.errors()[0]["type"] == "graph.maximum_cut.vertex_bound"


def test_vertex_map_request_preserves_rows_and_canonical_producer_consumer_handoff() -> (
    None
):
    vertex_map: dict[str, Any] = {
        "source_graph": {"vertices": ["b", "a"], "edges": [["b", "a"]]},
        "target_graph": {"vertices": ["y", "x"], "edges": [["y", "x"]]},
        "rows": [
            {"source_vertex": "a", "target_vertex": "x"},
            {"source_vertex": "b", "target_vertex": "y"},
        ],
    }
    operation = "graph.homomorphism.check"
    output = invoke_operation(
        operation, {"vertex_map": vertex_map}, Catalog.open()
    ).output
    assert output["status"] == "HOMOMORPHISM"
    produced = output["homomorphism"]["vertex_map"]
    assert produced["rows"] == vertex_map["rows"]
    assert produced["source_graph"]["vertices"] == ["b", "a"]
    assert produced["target_graph"]["vertices"] == ["y", "x"]
    assert (
        GraphVertexMap.model_validate_json(
            json.dumps(produced), strict=True
        ).model_dump(mode="json")
        == produced
    )
    assert (
        invoke_operation(operation, {"vertex_map": produced}, Catalog.open()).output
        == output
    )
    with pytest.raises(ValidationError):
        GraphVertexMap.model_validate_json(json.dumps(vertex_map), strict=True)
    collapsed = copy.deepcopy(vertex_map)
    collapsed["rows"][1]["target_vertex"] = "x"
    assert (
        invoke_operation(operation, {"vertex_map": collapsed}, Catalog.open()).output[
            "status"
        ]
        == "EDGE_IMAGE_NOT_EDGE"
    )


def test_signed_embedding_pair_reversal_preserves_rotations_and_twist_parity() -> None:
    operation = Catalog.open().operation("graph.embedding.nonorientable.check")
    assert operation is not None
    graph: dict[str, Any] = {
        "vertices": ["c", "a", "b"],
        "edges": [["a", "c"], ["a", "b"], ["b", "c"]],
    }
    payload: dict[str, Any] = {
        "graph": graph,
        "rotations": [[0, 2], [0, 1], [1, 2]],
        "signs": [0, 1, 1],
    }
    canonical = invoke_operation(operation.operation_id, payload, Catalog.open()).output
    reversed_payload = copy.deepcopy(payload)
    reversed_payload["graph"]["edges"] = [
        list(reversed(edge)) for edge in graph["edges"]
    ]
    normalized = invoke_operation(
        operation.operation_id, reversed_payload, Catalog.open()
    ).output
    assert normalized == canonical
    assert normalized["graph"] == graph
    assert normalized["rotations"] == payload["rotations"]
    assert normalized["signs"] == payload["signs"]
    assert normalized["orientable"] is False
    assert (
        operation.result_type.model_validate_json(
            json.dumps(normalized), strict=True
        ).model_dump(mode="json")
        == normalized
    )
    corrupted = copy.deepcopy(normalized)
    corrupted["graph"]["edges"][0].reverse()
    with pytest.raises(ValidationError) as error:
        operation.result_type.model_validate_json(json.dumps(corrupted), strict=True)
    assert error.value.errors()[0]["loc"] == ("graph",)
