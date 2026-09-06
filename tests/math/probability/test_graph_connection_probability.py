"""Exact undirected terminal-reliability operation contracts."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.probability._graph_connection_probability import (
    GRAPH_CONNECTION_PROBABILITY_OPERATION,
    GraphConnectionProbabilityRequest,
    GraphReliabilityEdgeProbability,
    verify_graph_connection_probability,
)


def test_operation_preserves_the_complete_triangle_ledger() -> None:
    """The catalog operation still parses and projects its canonical wire result."""

    operation = GRAPH_CONNECTION_PROBABILITY_OPERATION
    payload = operation.examples[0].input

    request = operation.request_type.model_validate(payload)
    result = operation.run(request)
    reparsed = operation.result_type.model_validate(result.model_dump(mode="json"))

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
                edge=("a", "b"), open_probability={"num": "2", "den": "1"}
            ),
        ),
        terminals=("a", "b"),
    )

    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        GRAPH_CONNECTION_PROBABILITY_OPERATION.run(request)
