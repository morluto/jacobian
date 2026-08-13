from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from jacobian.checker_identity import build_checker_manifest
from jacobian.contracts.verification import VerificationRecord

_ARTIFACT_URI = "artifact://sha256/" + "a" * 64
_CHECKER_URI = "checker://sha256/" + "b" * 64
_DIGEST = "sha256:" + "c" * 64


def _record(**updates: object) -> dict[str, object]:
    manifest = build_checker_manifest(
        "jacobian_checkers.reject:check",
        provider_runtime=None,
        passive_contract_uris=(),
    )
    payload: dict[str, object] = {
        "record_schema_version": "4",
        "checker_id": _CHECKER_URI,
        "implementation_digest": manifest.implementation_digest(),
        "checker_manifest": manifest.model_dump(mode="json"),
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


def test_v4_verification_record_requires_its_exact_checker_manifest() -> None:
    payload = _record()
    record = VerificationRecord.model_validate(payload)
    assert (
        record.implementation_digest == record.checker_manifest.implementation_digest()
    )

    with pytest.raises(ValidationError):
        VerificationRecord.model_validate(payload | {"checker_manifest": None})

    with pytest.raises(ValidationError, match="digest must match its manifest"):
        VerificationRecord.model_validate(
            payload | {"implementation_digest": "sha256:" + "0" * 64}
        )


def test_v4_verification_record_rejects_a_mutated_manifest_snapshot() -> None:
    payload = _record()
    manifest = dict(cast(dict[str, object], payload["checker_manifest"]))
    sandbox = dict(cast(dict[str, object], manifest["sandbox"]))
    sandbox["max_wall_seconds"] = cast(int, sandbox["max_wall_seconds"]) + 1
    sandbox["max_cpu_seconds"] = cast(int, sandbox["max_cpu_seconds"]) + 1
    manifest["sandbox"] = sandbox

    with pytest.raises(ValidationError, match="digest must match its manifest"):
        VerificationRecord.model_validate(payload | {"checker_manifest": manifest})


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
        VerificationRecord.model_validate(_record() | {field: value})


def test_verification_record_requires_complete_unique_relationship_endpoints() -> None:
    with pytest.raises(ValidationError, match="requires exact endpoints"):
        VerificationRecord.model_validate(
            _record()
            | {
                "relation_id": "graph.isomorphism.relation",
                "relationship_source_artifact_uris": (_ARTIFACT_URI,),
            }
        )

    with pytest.raises(ValidationError, match="must be unique"):
        VerificationRecord.model_validate(
            _record()
            | {
                "relation_id": "graph.isomorphism.relation",
                "relationship_source_artifact_uris": (
                    _ARTIFACT_URI,
                    _ARTIFACT_URI,
                ),
                "relationship_target_artifact_uris": (_ARTIFACT_URI,),
            }
        )
