from __future__ import annotations

import pytest
from tests.support.services import open_reference_services

from jacobian.artifacts import ArtifactValidationError


def _claim_payload(
    *,
    domain_id: str,
    semantics_uri: str,
    predicate: str,
    parameters: dict[str, object],
) -> dict[str, object]:
    return {
        "claim_schema_version": "1",
        "domain_id": domain_id,
        "domain_version": "1",
        "semantics_uri": semantics_uri,
        "quantifiers": [],
        "predicate": {
            "name": predicate,
            "parameters": parameters,
        },
        "bounds": {},
        "required_capabilities": [],
        "correspondence_status": "UNREVIEWED",
    }


def test_reference_schemas_reject_incomplete_claims_and_candidates(tmp_path) -> None:
    with open_reference_services(
        tmp_path / "state",
        "graph_paths",
        "matrices",
        "erdos_straus",
    ) as services:
        graph = services.references["graph_paths"]
        with pytest.raises(ArtifactValidationError, match="simple"):
            services.core.artifacts.put(
                schema_uri=graph.claim_schema_uri,
                semantics_uri=graph.semantics_uri,
                payload=_claim_payload(
                    domain_id="jacobian.graph-paths",
                    semantics_uri=graph.semantics_uri,
                    predicate="intended_paths_complete",
                    parameters={},
                ),
            )

        matrix = services.references["matrices"]
        with pytest.raises(ArtifactValidationError, match="scope"):
            services.core.artifacts.put(
                schema_uri=matrix.claim_schema_uri,
                semantics_uri=matrix.semantics_uri,
                payload=_claim_payload(
                    domain_id="jacobian.integer-matrices",
                    semantics_uri=matrix.semantics_uri,
                    predicate="maximize_absolute_determinant",
                    parameters={},
                ),
            )

        with pytest.raises(ArtifactValidationError):
            services.core.artifacts.put(
                schema_uri=graph.candidate_schema_uri,
                semantics_uri=graph.semantics_uri,
                payload={
                    "vertices": ["s", "t"],
                    "arcs": [["s"]],
                },
            )

        erdos_straus = services.references["erdos_straus"]
        with pytest.raises(ArtifactValidationError, match="upper_bound"):
            services.core.artifacts.put(
                schema_uri=erdos_straus.claim_schema_uri,
                semantics_uri=erdos_straus.semantics_uri,
                payload=_claim_payload(
                    domain_id="jacobian.erdos-straus",
                    semantics_uri=erdos_straus.semantics_uri,
                    predicate="erdos_straus_range",
                    parameters={"lower_bound": 2},
                ),
            )
