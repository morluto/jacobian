from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graph_coloring_ops._models import (
    KColorabilityRequest,
    MaximumIndependentSetRequest,
)
from jacobian.math.graph_coloring_ops._operations import (
    compute_k_colorability,
    compute_maximum_independent_set,
)
from jacobian.math.graph_flow._models import MaxFlowRequest
from jacobian.math.graph_flow._operations import compute_max_flow
from jacobian.math.graph_spectral._models import GraphSpectrumRequest
from jacobian.math.graph_spectral._operations import compute_laplacian_spectrum


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


def test_maximum_independent_set_is_exact() -> None:
    request = MaximumIndependentSetRequest.model_validate(
        {
            "graph": {
                "vertex_count": 5,
                "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
            }
        }
    )

    result = compute_maximum_independent_set(request)

    assert result.cardinality == 2
    assert all(
        left not in result.independent_set or right not in result.independent_set
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
