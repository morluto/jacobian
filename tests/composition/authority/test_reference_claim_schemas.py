from __future__ import annotations

import pytest

from jacobian.artifacts import ArtifactValidationError

pytestmark = pytest.mark.xdist_group("reference-claim-schemas")


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


def test_path_closure_claim_requires_simple_path_semantics(
    authorized_complete_runtime,
) -> None:
    reference = authorized_complete_runtime.portfolio.references["graph_paths"]

    with pytest.raises(ArtifactValidationError, match="simple"):
        authorized_complete_runtime.core.artifacts.put(
            schema_uri=reference.claim_schema_uri,
            semantics_uri=reference.semantics_uri,
            payload=_claim_payload(
                domain_id="jacobian.graph-paths",
                semantics_uri=reference.semantics_uri,
                predicate="intended_paths_complete",
                parameters={},
            ),
        )


def test_maxdet_claim_requires_a_bounded_matrix_scope(
    authorized_complete_runtime,
) -> None:
    reference = authorized_complete_runtime.portfolio.references["matrices"]

    with pytest.raises(ArtifactValidationError, match="scope"):
        authorized_complete_runtime.core.artifacts.put(
            schema_uri=reference.claim_schema_uri,
            semantics_uri=reference.semantics_uri,
            payload=_claim_payload(
                domain_id="jacobian.integer-matrices",
                semantics_uri=reference.semantics_uri,
                predicate="maximize_absolute_determinant",
                parameters={},
            ),
        )


def test_graph_candidate_schema_rejects_incomplete_arc(
    authorized_complete_runtime,
) -> None:
    reference = authorized_complete_runtime.portfolio.references["graph_paths"]

    with pytest.raises(ArtifactValidationError):
        authorized_complete_runtime.core.artifacts.put(
            schema_uri=reference.candidate_schema_uri,
            semantics_uri=reference.semantics_uri,
            payload={
                "vertices": ["s", "t"],
                "arcs": [["s"]],
            },
        )


def test_erdos_straus_claim_requires_a_bounded_range(
    authorized_complete_runtime,
) -> None:
    reference = authorized_complete_runtime.portfolio.references["erdos_straus"]

    with pytest.raises(ArtifactValidationError, match="upper_bound"):
        authorized_complete_runtime.core.artifacts.put(
            schema_uri=reference.claim_schema_uri,
            semantics_uri=reference.semantics_uri,
            payload=_claim_payload(
                domain_id="jacobian.erdos-straus",
                semantics_uri=reference.semantics_uri,
                predicate="erdos_straus_range",
                parameters={"lower_bound": 2},
            ),
        )
