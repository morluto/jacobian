from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDiscoveryRequest
from jacobian.math.graphs.coloring._admission import ADMISSIONS
from jacobian.math.graphs.coloring._models import (
    KColorabilityRequest,
)
from jacobian.math.graphs.coloring._operations import (
    compute_k_colorability,
)
from jacobian.math.graphs.flow._models import MaxFlowRequest
from jacobian.math.graphs.flow._operations import compute_max_flow
from jacobian.math.graphs.independence import (
    IndependenceNumberRequest,
    independence_number,
)
from jacobian.math.graphs.spectral._models import GraphSpectrumRequest
from jacobian.math.graphs.spectral._operations import compute_laplacian_spectrum


def test_k_colorability_uses_an_exact_decision_procedure() -> None:
    request = KColorabilityRequest.model_validate(
        {
            "graph": {
                "vertex_count": 6,
                "edges": [
                    [0, 3],
                    [0, 4],
                    [0, 5],
                    [1, 3],
                    [1, 4],
                    [1, 5],
                    [2, 3],
                    [2, 4],
                    [2, 5],
                ],
            },
            "colors": 2,
        }
    )

    result = compute_k_colorability(request)

    assert result.colorable is True
    assert result.coloring is not None
    assert all(
        result.coloring[left] != result.coloring[right]
        for left, right in request.graph.edges
    )


def test_flow_preserves_large_exact_rational_capacity() -> None:
    result = compute_max_flow(
        MaxFlowRequest.model_validate(
            {
                "graph": {
                    "vertex_count": 2,
                    "edges": [
                        {
                            "source": 0,
                            "target": 1,
                            "capacity": {"num": "9007199254740993", "den": "1"},
                        }
                    ],
                },
                "source": 0,
                "sink": 1,
            }
        )
    )

    assert result.flow_value.num == "9007199254740993"
    assert result.flow_value.den == "1"


def test_flow_contract_rejects_out_of_range_terminals() -> None:
    with pytest.raises(ValidationError, match="source must be"):
        MaxFlowRequest.model_validate(
            {
                "graph": {
                    "vertex_count": 2,
                    "edges": [
                        {
                            "source": 0,
                            "target": 1,
                            "capacity": {"num": "1", "den": "1"},
                        }
                    ],
                },
                "source": 2,
                "sink": 1,
            }
        )


def test_spectral_contract_rejects_non_simple_graphs() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        GraphSpectrumRequest.model_validate(
            {"graph": {"vertex_count": 2, "edges": [[0, 1], [1, 0]]}}
        )


def test_native_spectral_api_requires_a_validated_simple_graph() -> None:
    from jacobian.math.graphs.spectral import (
        GraphEdgeList,
        adjacency_spectrum,
        laplacian_spectrum,
    )

    graph = GraphEdgeList(vertex_count=2, edges=((0, 1),))
    assert dict(laplacian_spectrum(graph)) == {"0": 1, "2": 1}

    with pytest.raises(ValidationError, match="self-loops"):
        adjacency_spectrum(GraphEdgeList(vertex_count=2, edges=((0, 0),)))


def test_laplacian_spectrum_uses_normalized_simple_graph_degree() -> None:
    result = compute_laplacian_spectrum(
        GraphSpectrumRequest.model_validate(
            {"graph": {"vertex_count": 2, "edges": [[0, 1]]}}
        )
    )

    assert dict(zip(result.eigenvalues, result.multiplicities, strict=True)) == {
        "0": 1,
        "2": 1,
    }


def test_catalog_retires_the_duplicate_and_discovers_independence_number() -> None:
    retired_operation_id = "graph.independent_set.maximum.compute"
    catalog = Catalog.open()
    assert catalog.operation(retired_operation_id) is None
    assert retired_operation_id not in {
        operation.operation_id
        for operation in catalog.browse(
            domain="graph", limit=20, cursor=None
        ).operations
    }
    assert retired_operation_id not in {
        admission.operation_id for admission in ADMISSIONS
    }

    discovered = catalog.search(
        OperationDiscoveryRequest(query="maximum independent set", limit=5)
    )
    discovered_ids = {match.operation_id for match in discovered.matches}
    assert retired_operation_id not in discovered_ids
    assert "graph.invariant.independence_number.compute" in discovered_ids


def test_exact_independence_witness_is_independent_and_binds_its_bounds() -> None:
    request = IndependenceNumberRequest.model_validate(
        {
            "graph": {
                "vertices": ["0", "1", "2", "3", "4"],
                "edges": [
                    ["0", "1"],
                    ["1", "2"],
                    ["2", "3"],
                    ["3", "4"],
                    ["0", "4"],
                ],
            },
            "resource_budget": {"wall_seconds": 5, "max_order": 5},
        }
    )
    result = independence_number(request)

    assert result.status == "EXACT"
    assert result.optimum_value == result.incumbent_value == result.lower_bound
    assert result.upper_bound == result.optimum_value
    witness = set(result.witness_vertices)
    assert all(
        left not in witness or right not in witness
        for left, right in request.graph.edges
    )
