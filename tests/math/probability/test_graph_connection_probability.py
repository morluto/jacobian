"""Exact undirected terminal-reliability operation contracts."""

from __future__ import annotations

from fractions import Fraction

from jacobian.math.probability._graph_connection_probability import (
    GRAPH_CONNECTION_PROBABILITY_OPERATION,
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
