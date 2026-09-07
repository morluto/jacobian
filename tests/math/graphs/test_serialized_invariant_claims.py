"""Graph-relative claims require explicit relation checking."""

import json
from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.chordal import recognize_chordal, verify_chordality
from jacobian.math.graphs.chordal._models import ChordalRecognitionResult
from jacobian.math.graphs.optimization import (
    verify_graph_invariant,
    verify_maximum_matching,
)
from jacobian.math.graphs.optimization._invariant_models import (
    GraphCoreResult,
    GraphGirthResult,
    GraphMaximumMatchingRequest,
    GraphMaximumMatchingResult,
)
from jacobian.math.graphs.optimization._invariants import _maximum_matching_execute
from jacobian.math.graphs.values import SimpleUndirectedGraph


def test_chordality_checks_chords_and_later_neighbors() -> None:
    vertices = ("a", "b", "c", "d")
    complete = SimpleUndirectedGraph(
        vertices=vertices, edges=tuple(combinations(vertices, 2))
    )
    cycle = SimpleUndirectedGraph(
        vertices=vertices, edges=(("a", "b"), ("a", "d"), ("b", "c"), ("c", "d"))
    )
    forged_cycle = ChordalRecognitionResult(
        graph=complete, status="NONCHORDAL", induced_cycle=vertices
    )
    forged_order = ChordalRecognitionResult(
        graph=cycle, status="CHORDAL", elimination_ordering=vertices
    )
    for claim in (forged_cycle, forged_order):
        assert not verify_chordality(
            ChordalRecognitionResult.model_validate_json(claim.model_dump_json())
        )
    for graph in (complete, cycle, SimpleUndirectedGraph(vertices=(), edges=())):
        result = recognize_chordal(graph)
        assert verify_chordality(
            ChordalRecognitionResult.model_validate_json(result.model_dump_json())
        )


@pytest.mark.parametrize("change", ["edge", "barrier", "odd_components"])
def test_matching_certificate_checks_source_relation(change: str) -> None:
    graph = SimpleUndirectedGraph(vertices=("a", "b", "c"), edges=(("a", "b"),))
    result = _maximum_matching_execute(GraphMaximumMatchingRequest(graph=graph))
    payload = result.model_dump(mode="json")
    assert verify_maximum_matching(
        GraphMaximumMatchingResult.model_validate_json(json.dumps(payload))
    )
    if change == "edge":
        payload["witness_edges"] = [["a", "c"]]
    elif change == "barrier":
        payload["certificate"]["barrier_vertices"] = ["outside"]
    else:
        payload["certificate"]["odd_component_count"] = 0
    assert not verify_maximum_matching(
        GraphMaximumMatchingResult.model_validate_json(json.dumps(payload))
    )


def test_graph_invariant_and_core_keep_source() -> None:
    graph = SimpleUndirectedGraph(vertices=("a",), edges=())
    forged = GraphGirthResult(graph=graph, girth=3, has_cycle=True)
    assert not verify_graph_invariant(
        GraphGirthResult.model_validate_json(forged.model_dump_json())
    )
    valid = GraphCoreResult(graph=graph, k=1, vertices=())
    assert verify_graph_invariant(
        GraphCoreResult.model_validate_json(valid.model_dump_json())
    )
    with pytest.raises(ValidationError, match="source graph"):
        GraphCoreResult(graph=graph, k=0, vertices=("outside",))
