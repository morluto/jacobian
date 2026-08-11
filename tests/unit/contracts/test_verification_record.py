from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.verification import VerificationRecord

_ARTIFACT_URI = "artifact://sha256/" + "a" * 64
_CHECKER_URI = "checker://sha256/" + "b" * 64
_DIGEST = "sha256:" + "c" * 64


def _record(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "checker_id": _CHECKER_URI,
        "checker_digest": _DIGEST,
        "evidence_kind": "WITNESS",
        "evidence_uri": _ARTIFACT_URI,
        "bindings": {
            "claim_digest": _DIGEST,
            "semantics_digest": _DIGEST,
            "candidate_digest": _DIGEST,
        },
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "request_digest": _DIGEST,
        "environment_digest": _DIGEST,
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("conclusion", "UNKNOWN"),
        ("arithmetic", "FLOATING_HEURISTIC"),
        ("coverage", "SAMPLED"),
        ("coverage", "EXHAUSTIVE"),
    ],
)
def test_verification_record_rejects_non_replayable_evidence(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValidationError):
        VerificationRecord.model_validate(_record(**{field: value}))


def test_verification_record_requires_complete_unique_relationship_endpoints() -> None:
    with pytest.raises(ValidationError, match="requires exact endpoints"):
        VerificationRecord.model_validate(
            _record(
                relation_id="graph.isomorphism.relation",
                relationship_source_artifact_uris=(_ARTIFACT_URI,),
            )
        )

    with pytest.raises(ValidationError, match="must be unique"):
        VerificationRecord.model_validate(
            _record(
                relation_id="graph.isomorphism.relation",
                relationship_source_artifact_uris=(_ARTIFACT_URI, _ARTIFACT_URI),
                relationship_target_artifact_uris=(_ARTIFACT_URI,),
            )
        )
