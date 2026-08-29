"""Owner operations reject schema-valid semantic mutations with typed errors."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis.boolean import walsh_hadamard_transform
from jacobian.math.graphs._tools import TOOLS as GRAPH_TOOLS
from jacobian.math.graphs._tools import Graph6DecodeRequest
from jacobian.math.probability._graph_connection_probability import (
    GraphConnectionProbabilityRequest,
    compute_graph_connection_probability,
)


def test_graph_connection_probability_rejects_unbound_edge_probabilities() -> None:
    request = GraphConnectionProbabilityRequest.model_validate(
        {
            "graph": {
                "vertices": ["a", "b", "c"],
                "edges": [],
            },
            "edge_probabilities": [
                {"edge": ["a", "b"], "open_probability": {"num": "1", "den": "2"}},
                {"edge": ["a", "c"], "open_probability": {"num": "1", "den": "2"}},
                {"edge": ["b", "c"], "open_probability": {"num": "1", "den": "2"}},
            ],
            "terminals": ["a", "c"],
        }
    )

    with pytest.raises(OperationDomainValidationError) as error:
        compute_graph_connection_probability(request)

    assert error.value.errors()[0]["loc"] == ("edge_probabilities",)


def test_walsh_transform_rejects_non_power_of_two_truth_table() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        walsh_hadamard_transform((0, 0, 0))

    assert error.value.errors()[0]["loc"] == ("truth_table",)


def test_graph6_operation_rejects_malformed_payload() -> None:
    request = Graph6DecodeRequest(graph6=" ")

    with pytest.raises(OperationDomainValidationError) as error:
        GRAPH_TOOLS[0].run(request)

    assert error.value.errors()[0]["loc"] == ("graph6",)
