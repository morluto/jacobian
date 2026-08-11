from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.plugin_graphs import (
    GraphPathCapabilityRequest,
    GraphPathReductionRequest,
    GraphShrinkRequest,
)
from jacobian.contracts.plugin_matrices import (
    MatrixCapabilityRequest,
    MatrixReductionRequest,
    MatrixTransformRequest,
)
from jacobian.contracts.plugin_number_theory import ErdosStrausCapabilityRequest
from jacobian.contracts.shrinking import PluginReductionResponse
from jacobian.plugins.graph_shrinking import reduce_simple_graph


def _matrix_candidate() -> dict[str, object]:
    return {"rows": 1, "cols": 1, "entries": [["1"]]}


def test_plugin_contracts_reject_non_object_nested_payloads() -> None:
    with pytest.raises(ValidationError):
        GraphPathReductionRequest.model_validate(
            {
                "target": "a",
                "claim": {"predicate": "is_bipartite"},
            }
        )
    with pytest.raises(ValidationError):
        MatrixReductionRequest.model_validate(
            {
                "target": _matrix_candidate(),
                "claim": "is_nonsingular",
            }
        )


def test_plugin_contracts_reject_scalar_collections_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        GraphShrinkRequest.model_validate(
            {
                "target": {"vertices": ["a"], "edges": []},
                "reducers": "delete_vertex",
                "objectives": [],
            }
        )
    with pytest.raises(ValidationError):
        MatrixReductionRequest.model_validate(
            {
                "target": _matrix_candidate(),
                "claim": {"predicate": "is_nonsingular"},
                "reducers": [],
                "objectives": [],
                "unexpected": True,
            }
        )


def test_graph_shrinker_returns_schema_valid_typed_graph_proposals() -> None:
    artifact_uri = "artifact://sha256/" + "a" * 64
    response = PluginReductionResponse.model_validate(
        reduce_simple_graph(
            {
                "request_version": "1",
                "target": {
                    "graph_schema_version": "1",
                    "vertices": ["a", "b"],
                    "edges": [["a", "b"]],
                },
                "claim": {
                    "domain_id": "jacobian.graph-shrinking",
                    "domain_version": "1",
                    "semantics_uri": artifact_uri,
                    "predicate": {"name": "graph.property.non_bipartite"},
                    "required_capabilities": ["Reducer"],
                    "correspondence_status": "FORMALLY_LINKED",
                },
                "reducers": ["delete_vertex", "delete_edge"],
                "objectives": ["vertices", "edges"],
            }
        )
    )

    assert len(response.reductions) == 3
    assert response.reductions[0].payload == {
        "graph_schema_version": "1",
        "vertices": ["b"],
        "edges": [],
    }


def test_plugin_contracts_accept_known_worker_metadata_without_opening_the_boundary() -> (
    None
):
    request = GraphPathCapabilityRequest.model_validate(
        {
            "request_version": "1",
            "profile": "FAST",
            "seed": 0,
            "bindings": {"claim_digest": "sha256:" + "a" * 64},
            "claim": {"predicate": "is_bipartite"},
            "candidate": {
                "vertices": ["a", "b"],
                "arcs": [["a", "b"]],
            },
        }
    )
    assert request.request_version == "1"
    assert request.profile == "FAST"
    assert request.seed == 0

    with pytest.raises(ValidationError):
        MatrixTransformRequest.model_validate(
            {
                "request_version": "1",
                "source": _matrix_candidate(),
                "unexpected": True,
            }
        )


def test_erdos_straus_contract_enforces_range_and_role_domain() -> None:
    with pytest.raises(ValidationError):
        ErdosStrausCapabilityRequest.model_validate(
            {
                "claim": {
                    "predicate": "erdos_straus_range",
                    "lower_bound": 100,
                    "upper_bound": 2,
                },
                "candidate": {"lower_bound": 100, "upper_bound": 2},
            }
        )


def test_nested_matrix_scope_is_typed_and_bounded() -> None:
    with pytest.raises(ValidationError):
        MatrixCapabilityRequest.model_validate(
            {
                "claim": {
                    "predicate": "maximize_absolute_determinant",
                    "scope": {
                        "rows": 2,
                        "cols": 3,
                        "entries": [-1, 1],
                        "unexpected": True,
                    },
                }
            }
        )
    with pytest.raises(ValidationError):
        MatrixCapabilityRequest.model_validate(
            {
                "claim": {
                    "predicate": "maximize_absolute_determinant",
                    "scope": {"rows": 2, "cols": 3, "entries": [-1, 1]},
                }
            }
        )
    with pytest.raises(ValidationError):
        ErdosStrausCapabilityRequest.model_validate(
            {
                "claim": {
                    "predicate": "erdos_straus_range",
                    "lower_bound": 2,
                    "upper_bound": 3,
                },
                "candidate": {"lower_bound": 2, "upper_bound": 3},
                "witness_role": "unsupported",
            }
        )


def test_matrix_candidate_rejects_noncanonical_integer_strings() -> None:
    with pytest.raises(ValidationError):
        MatrixCapabilityRequest.model_validate(
            {
                "claim": {"predicate": "is_nonsingular"},
                "candidate": {"rows": 1, "cols": 1, "entries": [["01"]]},
            }
        )
