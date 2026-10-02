"""Graph-owned request spelling never changes the canonical value contract."""

import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from jacobian.math.graphs._input import (
    IndexedSimpleUndirectedGraphInput,
    LoopedSimpleGraphInput,
    SimpleUndirectedGraphInput,
)
from jacobian.math.graphs.directed._models import DirectedGraph
from jacobian.math.graphs.values import (
    MAX_ENCODED_SIMPLE_GRAPH_EDGES,
    MAX_ENCODED_SIMPLE_GRAPH_VERTICES,
    MAX_INDEXED_SIMPLE_GRAPH_EDGES,
    IndexedSimpleUndirectedGraph,
    LoopedSimpleGraph,
    SimpleUndirectedGraph,
)

_INPUT = TypeAdapter(SimpleUndirectedGraphInput)


@pytest.mark.parametrize(
    ("vertices", "edges", "expected"),
    (
        (["2", "10", "11"], [["2", "10"], ["11", "10"]], (("10", "2"), ("10", "11"))),
        (["é", "x", ""], [["é", "x"], ["x", ""]], (("x", "é"), ("", "x"))),
        ([], [], ()),
        (["alone"], [], ()),
        (["z" * 100, "a"], [["z" * 100, "a"]], (("a", "z" * 100),)),
    ),
)
def test_json_input_orients_pairs_and_preserves_each_axis(
    vertices: list[str], edges: list[list[str]], expected: tuple[tuple[str, str], ...]
) -> None:
    graph = _INPUT.validate_json(
        json.dumps({"vertices": vertices, "edges": edges}), strict=True
    )
    assert type(graph) is SimpleUndirectedGraph
    assert graph.vertices == tuple(vertices)
    assert graph.edges == expected
    assert (
        SimpleUndirectedGraph.model_validate_json(graph.model_dump_json(), strict=True)
        == graph
    )
    assert _INPUT.validate_json(graph.model_dump_json(), strict=True) == graph


def test_native_and_persisted_graph_decoding_remain_canonical() -> None:
    raw = {"vertices": ("z", "a"), "edges": (("z", "a"),)}
    decoders: tuple[Callable[[], Any], ...] = (
        lambda: _INPUT.validate_python(raw, strict=True),
        lambda: SimpleUndirectedGraph.model_validate(raw, strict=True),
        lambda: SimpleUndirectedGraph.model_validate_json(json.dumps(raw), strict=True),
    )
    for decode in decoders:
        with pytest.raises(ValidationError) as error:
            decode()
        assert (
            error.value.errors()[0]["type"]
            == "graph.edges_must_contain_two_declared_vertices_in_orde"
        )


@pytest.mark.parametrize(
    ("vertices", "edges", "code", "location"),
    (
        (["a", "b"], [["b", "a"], ["a", "b"]], "graph.graph_edges_must_be_unique", ()),
        (["a", "b"], [["b", "a"], ["b", "a"]], "graph.graph_edges_must_be_unique", ()),
        (
            ["a"],
            [["a", "a"]],
            "graph.edges_must_contain_two_declared_vertices_in_orde",
            (),
        ),
        (
            ["a"],
            [["z", "a"]],
            "graph.edges_must_contain_two_declared_vertices_in_orde",
            (),
        ),
        (["a", "a"], [], "graph.graph_vertices_must_be_unique", ()),
        (["e\u0301"], [], "graph.graph_vertices_must_use_unicode_nfc", ()),
        (["\ud800"], [], "json_invalid", ()),
        (["a", "b"], [[True, "a"]], "string_type", ("edges", 0, 0)),
        (["a", "b"], [[2, "a"]], "string_type", ("edges", 0, 0)),
        (["a", "b"], [["b"]], "missing", ("edges", 0, 1)),
        (["a", "b"], [["b", "a", "b"]], "too_long", ("edges", 0)),
    ),
)
def test_invalid_inputs_keep_canonical_invariants_and_shape_locations(
    vertices: list[str],
    edges: list[list[Any]],
    code: str,
    location: tuple[str | int, ...],
) -> None:
    with pytest.raises(ValidationError) as error:
        _INPUT.validate_json(
            json.dumps({"vertices": vertices, "edges": edges}), strict=True
        )
    assert error.value.errors()[0]["type"] == code
    assert error.value.errors()[0]["loc"] == location


def test_raw_edge_count_is_checked_before_orientation_and_duplicate_detection() -> None:
    raw = {
        "vertices": ["a", "b"],
        "edges": [["b", "a"]] * (MAX_ENCODED_SIMPLE_GRAPH_EDGES + 1),
    }
    with pytest.raises(ValidationError) as error:
        _INPUT.validate_json(json.dumps(raw), strict=True)
    assert error.value.errors()[0]["type"] == "too_long"
    assert error.value.errors()[0]["loc"] == ("edges",)


def test_raw_vertex_count_keeps_the_carrier_envelope() -> None:
    raw = {
        "vertices": [
            str(index) for index in range(MAX_ENCODED_SIMPLE_GRAPH_VERTICES + 1)
        ],
        "edges": [],
    }
    with pytest.raises(ValidationError) as error:
        _INPUT.validate_json(json.dumps(raw), strict=True)
    assert error.value.errors()[0]["type"] == "too_long"
    assert error.value.errors()[0]["loc"] == ("vertices",)


def test_directed_and_indexed_values_are_not_reinterpreted() -> None:
    directed = DirectedGraph.model_validate_json(
        '{"vertex_count":2,"edges":[[1,0]]}', strict=True
    )
    assert directed.edges == ((1, 0),)
    with pytest.raises(ValidationError) as error:
        IndexedSimpleUndirectedGraph.model_validate_json(
            '{"vertex_count":2,"edges":[[1,0]]}', strict=True
        )
    assert (
        error.value.errors()[0]["type"]
        == "graph.indexed_edges_must_be_canonical_pairs_with_left"
    )


def test_indexed_input_uses_numeric_pair_order_and_keeps_edge_axis() -> None:
    adapter = TypeAdapter(IndexedSimpleUndirectedGraphInput)
    raw = {"vertex_count": 12, "edges": [[10, 2], [11, 10]]}
    graph = adapter.validate_json(json.dumps(raw), strict=True)
    assert type(graph) is IndexedSimpleUndirectedGraph
    assert graph.edges == ((2, 10), (10, 11))
    assert (
        IndexedSimpleUndirectedGraph.model_validate_json(
            graph.model_dump_json(), strict=True
        )
        == graph
    )
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {"vertex_count": 12, "edges": ((10, 2), (11, 10))}, strict=True
        )


@pytest.mark.parametrize(
    ("edges", "code"),
    (
        ([[1, 0], [0, 1]], "graph.a_simple_graph_cannot_contain_duplicate_edges"),
        ([[1, 1]], "graph.a_simple_graph_cannot_contain_self_loops"),
        ([[2, 0]], "graph.edge_vertices_must_be_in_0_vertex_count_1"),
        ([[0, -1]], "graph.edge_vertices_must_be_in_0_vertex_count_1"),
        ([[True, 0]], "int_type"),
        ([[1.0, 0]], "int_type"),
        ([["1", 0]], "int_type"),
    ),
)
def test_indexed_input_rejects_invalid_edges_and_noninteger_indices(
    edges: list[list[Any]], code: str
) -> None:
    with pytest.raises(ValidationError) as error:
        TypeAdapter(IndexedSimpleUndirectedGraphInput).validate_json(
            json.dumps({"vertex_count": 2, "edges": edges}), strict=True
        )
    assert error.value.errors()[0]["type"] == code


def test_indexed_raw_count_precedes_normalization_and_duplicates() -> None:
    with pytest.raises(ValidationError) as error:
        TypeAdapter(IndexedSimpleUndirectedGraphInput).validate_json(
            json.dumps(
                {
                    "vertex_count": 2,
                    "edges": [[1, 0]] * (MAX_INDEXED_SIMPLE_GRAPH_EDGES + 1),
                }
            ),
            strict=True,
        )
    assert error.value.errors()[0]["type"] == "too_long"
    assert error.value.errors()[0]["loc"] == ("edges",)


def test_looped_graph_input_preserves_loop_and_vertex_axes() -> None:
    raw = {"vertices": ["z", "a"], "edges": [["z", "a"]], "loops": ["z", "a"]}
    graph = TypeAdapter(LoopedSimpleGraphInput).validate_json(
        json.dumps(raw), strict=True
    )
    assert type(graph) is LoopedSimpleGraph
    assert graph.vertices == ("z", "a")
    assert graph.edges == (("a", "z"),)
    assert graph.loops == ("z", "a")
    assert (
        LoopedSimpleGraph.model_validate_json(graph.model_dump_json(), strict=True)
        == graph
    )
    with pytest.raises(ValidationError):
        LoopedSimpleGraph.model_validate_json(json.dumps(raw), strict=True)


@pytest.mark.parametrize(
    ("edges", "loops", "code"),
    (
        ([["z", "a"], ["a", "z"]], [], "graph.looped_edges_invalid"),
        ([["z", "z"]], [], "graph.looped_edges_invalid"),
        ([["z", "a"]], ["z", "z"], "graph.looped_vertices_invalid"),
        ([["z", "a"]], ["outside"], "graph.looped_vertices_invalid"),
    ),
)
def test_looped_input_preserves_edge_and_loop_invariants(
    edges: list[list[str]], loops: list[str], code: str
) -> None:
    with pytest.raises(ValidationError) as error:
        TypeAdapter(LoopedSimpleGraphInput).validate_json(
            json.dumps({"vertices": ["z", "a"], "edges": edges, "loops": loops}),
            strict=True,
        )
    assert error.value.errors()[0]["type"] == code
