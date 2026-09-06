"""Tree profiles retain their graph, bag-node and edge axes."""

import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.decomposition.tree_decompositions import (
    TreeDecomposition,
    adhesions,
    bag_intersection_graph,
    reroot,
    restrict,
    verify_adhesions,
    verify_bag_intersection_graph,
    verify_reroot,
    verify_vertex_occurrences,
    verify_width,
    vertex_occurrences,
    width,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def test_empty_graph_and_bag_profiles_compose() -> None:
    source = TreeDecomposition(
        graph=SimpleUndirectedGraph(vertices=("a",), edges=()),
        tree_nodes=("bag",),
        tree_edges=(),
        bags=(("a",),),
    )
    empty = restrict(source, frozenset())
    decoded = TreeDecomposition.model_validate_json(empty.model_dump_json())
    assert decoded.bags == ((),)
    assert width(decoded).width == -1
    assert vertex_occurrences(decoded).per_vertex == {}
    assert adhesions(decoded).edges == ()
    assert verify_width(width(decoded))
    assert verify_vertex_occurrences(vertex_occurrences(decoded))
    assert verify_adhesions(adhesions(decoded))
    assert verify_reroot(reroot(decoded, "bag"))
    assert verify_bag_intersection_graph(bag_intersection_graph(decoded))


def test_single_bag_empty_edge_profiles() -> None:
    source = TreeDecomposition(
        graph=SimpleUndirectedGraph(vertices=("a",), edges=()),
        tree_nodes=("bag",),
        tree_edges=(),
        bags=(("a",),),
    )
    result: Any
    verifier: Callable[[Any], bool]
    for result, verifier in (
        (width(source), verify_width),
        (reroot(source, "bag"), verify_reroot),
        (vertex_occurrences(source), verify_vertex_occurrences),
        (adhesions(source), verify_adhesions),
    ):
        assert result.decomposition == source
        assert verifier(type(result).model_validate_json(result.model_dump_json()))
    result = width(source)
    payload = result.model_dump()
    payload["width"] = 7
    assert not verify_width(type(result).model_validate_json(json.dumps(payload)))
    rooted = reroot(source, "bag")
    payload = rooted.model_dump()
    payload["root"] = "foreign"
    with pytest.raises(ValidationError):
        type(rooted).model_validate_json(json.dumps(payload))
    occurrence = vertex_occurrences(source)
    payload = occurrence.model_dump()
    payload["per_vertex"] = {"foreign": payload["per_vertex"]["a"]}
    with pytest.raises(ValidationError):
        type(occurrence).model_validate_json(json.dumps(payload))
