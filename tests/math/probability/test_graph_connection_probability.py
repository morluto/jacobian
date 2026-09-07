"""Exact undirected terminal-reliability operation contracts."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.probability._graph_connection_probability import (
    GRAPH_CONNECTION_PROBABILITY_OPERATION,
    GraphConnectionProbabilityRequest,
    GraphReliabilityEdgeProbability,
    GraphReliabilitySource,
    compute_graph_connection_probability,
    verify_graph_connection_probability,
)


def test_operation_preserves_the_complete_triangle_ledger() -> None:
    """The catalog operation still parses and projects its canonical wire result."""

    operation = GRAPH_CONNECTION_PROBABILITY_OPERATION
    payload = operation.examples[0].input

    request = operation.request_type.model_validate_json(json.dumps(payload))
    result = operation.run(request)
    reparsed = operation.result_type.model_validate_json(result.model_dump_json())

    assert operation.operation_id == (
        "probability.graph_reliability.connection_probability.compute"
    )
    assert result.connection_probability.as_fraction() == Fraction(5, 8)
    assert result.visited_states == 8
    assert reparsed == result
    assert result.source.graph == request.graph
    assert verify_graph_connection_probability(reparsed)
    assert not verify_graph_connection_probability(
        reparsed.model_copy(update={"connection_probability": {"num": "0", "den": "1"}})
    )


def test_probability_domain_is_admitted_at_operation_time() -> None:
    graph = SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),))
    request = GraphConnectionProbabilityRequest(
        graph=graph,
        edge_probabilities=(
            GraphReliabilityEdgeProbability(
                edge=("a", "b"), open_probability=CanonicalRational(num=2, den=1)
            ),
        ),
        terminals=("a", "b"),
    )

    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        GRAPH_CONNECTION_PROBABILITY_OPERATION.run(request)


def test_empty_edge_axis_is_retained_and_verifiable() -> None:
    request = GraphConnectionProbabilityRequest(
        graph=SimpleUndirectedGraph(vertices=("a", "b"), edges=()),
        edge_probabilities=(),
        terminals=("a", "b"),
    )
    result = compute_graph_connection_probability(
        GraphReliabilitySource.model_validate(request.model_dump())
    )

    assert result.connection_probability.as_fraction() == Fraction(0)
    assert result.edge_count == 0
    assert result.states[0].open_edge_indices == ()
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert decoded == result
    assert verify_graph_connection_probability(decoded)


def test_reliability_verifier_rejects_state_with_forged_edge_axis() -> None:
    request = GraphConnectionProbabilityRequest(
        graph=SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),)),
        edge_probabilities=(
            GraphReliabilityEdgeProbability(
                edge=("a", "b"), open_probability=CanonicalRational(num=1, den=2)
            ),
        ),
        terminals=("a", "b"),
    )
    result = compute_graph_connection_probability(
        GraphReliabilitySource.model_validate(request.model_dump())
    )
    decoded = type(result).model_validate_json(result.model_dump_json())
    forged_state = decoded.states[0].model_copy(update={"open_edge_indices": (0,)})
    forged = decoded.model_copy(update={"states": (forged_state, decoded.states[1])})

    assert not verify_graph_connection_probability(forged)
