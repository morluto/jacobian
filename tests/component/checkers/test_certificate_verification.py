from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.results import Conclusion, ExecutionStatus
from jacobian.registry import CheckerRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService


def _certificate_case(
    tmp_path: Path,
    *,
    checker_entrypoint: str = "jacobian_checkers.graph_paths:check_path_enumeration",
) -> tuple[ArtifactRepository, VerificationService, str]:
    store = ArtifactRepository(tmp_path)
    claim_schema = store.register_descriptor(
        kind="schema",
        name="graph.path-closure.claim",
        version="1",
        definition={"type": "object"},
    )
    candidate_schema = store.register_descriptor(
        kind="schema",
        name="graph.candidate",
        version="1",
        definition={"type": "object"},
    )
    scope_schema = store.register_descriptor(
        kind="schema",
        name="graph.path.scope",
        version="1",
        definition={"type": "object"},
    )
    certificate_schema = store.register_descriptor(
        kind="schema",
        name="graph.path-enumeration.certificate",
        version="1",
        definition=CertificateEnvelope.model_json_schema(),
    )
    semantics = store.register_descriptor(
        kind="semantics",
        name="directed-graph.path-language",
        version="1",
        definition={"paths": "all simple source-terminal paths"},
    )
    claim = store.put(
        schema_uri=claim_schema,
        semantics_uri=semantics,
        payload={"predicate": "intended_paths_complete", "simple": True},
    )
    candidate = store.put(
        schema_uri=candidate_schema,
        semantics_uri=semantics,
        payload={
            "vertices": ["s", "a", "b", "x", "t1", "t2"],
            "arcs": [
                ["s", "a"],
                ["a", "x"],
                ["s", "b"],
                ["b", "x"],
                ["x", "t1"],
                ["x", "t2"],
            ],
            "source": "s",
            "terminals": ["t1", "t2"],
            "intended_paths": [
                ["s", "a", "x", "t1"],
                ["s", "b", "x", "t2"],
            ],
        },
    )
    scope = store.put(
        schema_uri=scope_schema,
        semantics_uri=semantics,
        payload={"simple": True, "max_length": 6},
    )
    certificate_payload = {
        "actual_paths": [
            ["s", "a", "x", "t1"],
            ["s", "a", "x", "t2"],
            ["s", "b", "x", "t1"],
            ["s", "b", "x", "t2"],
        ]
    }
    certificate = CertificateEnvelope(
        certificate_type="graph.path_enumeration",
        format_version="1",
        bindings=EvidenceBindings(
            claim_digest=claim.object_digest,
            semantics_digest=store.get(semantics).manifest.object_digest,
            candidate_digest=candidate.object_digest,
            scope_digest=scope.object_digest,
        ),
        payload_digest="sha256:"
        + hashlib.sha256(canonicalize_json(certificate_payload)).hexdigest(),
        payload=certificate_payload,
    )
    certificate_artifact = store.put(
        schema_uri=certificate_schema,
        semantics_uri=semantics,
        payload=loads_strict_json(
            canonicalize_json(certificate.model_dump(mode="json"))
        ),
        parents=(
            claim.artifact_uri,
            candidate.artifact_uri,
            scope.artifact_uri,
        ),
    )
    registry = CheckerRegistry(store)
    registry.authorize(
        name="graph-path-enumeration-v1",
        entrypoint=checker_entrypoint,
        evidence_kind="CERTIFICATE",
        format_id="graph.path_enumeration",
        format_version="1",
        claim_schema_uris=(claim_schema,),
        semantics_uris=(semantics,),
        candidate_schema_uris=(candidate_schema,),
    )

    return (
        store,
        VerificationService(store, registry),
        certificate_artifact.artifact_uri,
    )


def test_complete_path_enumeration_certificate_is_verified(tmp_path: Path) -> None:
    _, service, certificate_uri = _certificate_case(tmp_path)

    result = service.verify_certificate(certificate_uri=certificate_uri)

    assert result.conclusion is Conclusion.FALSE
    assert result.verification_record_uri is not None


def test_certificate_supporting_artifacts_include_requested_storage_metadata(
    tmp_path: Path,
) -> None:
    store, service, certificate_uri = _certificate_case(
        tmp_path,
        checker_entrypoint=(
            "tests.component.checkers._fixture_checkers:"
            "check_supporting_artifact_metadata"
        ),
    )
    certificate = store.get(certificate_uri)
    supporting_schema = store.register_descriptor(
        kind="schema",
        name="graph.path-enumeration.supporting-evidence",
        version="1",
        definition={"type": "object"},
    )
    supporting_artifact = store.put(
        schema_uri=supporting_schema,
        semantics_uri=certificate.manifest.semantics_uri,
        payload={"kind": "supporting-evidence"},
    )

    result = service.verify_certificate(
        certificate_uri=certificate_uri,
        supporting_artifact_uris=(supporting_artifact.artifact_uri,),
        include_artifact_metadata=True,
    )

    assert result.conclusion is Conclusion.FALSE
    assert result.verification_record_uri is not None


def test_certificate_binding_substitution_is_rejected(tmp_path: Path) -> None:
    store, service, certificate_uri = _certificate_case(tmp_path)
    original = store.get(certificate_uri)
    payload = deepcopy(original.payload)
    payload["bindings"]["candidate_digest"] = "sha256:" + "9" * 64
    rebound = store.put(
        schema_uri=original.manifest.schema_uri,
        semantics_uri=original.manifest.semantics_uri,
        payload=payload,
        parents=original.manifest.parents,
    )

    result = service.verify_certificate(certificate_uri=rebound.artifact_uri)

    assert result.conclusion is Conclusion.UNKNOWN
    assert result.verification_record_uri is None
    assert result.verification_record_uri is None


def test_certificate_without_bound_parents_is_rejected(tmp_path: Path) -> None:
    store, service, certificate_uri = _certificate_case(tmp_path)
    original = store.get(certificate_uri)
    unbound = store.put(
        schema_uri=original.manifest.schema_uri,
        semantics_uri=original.manifest.semantics_uri,
        payload=original.payload,
    )

    result = service.verify_certificate(certificate_uri=unbound.artifact_uri)

    assert result.conclusion is Conclusion.UNKNOWN
    assert result.verification_record_uri is None
    assert result.verification_record_uri is None


def test_corrupt_certificate_payload_is_an_operational_failure(
    tmp_path: Path,
) -> None:
    store, service, certificate_uri = _certificate_case(tmp_path)
    certificate = store.get(certificate_uri)
    store._blob_path(certificate.manifest.payload_digest).write_bytes(b"corrupt")

    result = service.verify_certificate(certificate_uri=certificate_uri)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.conclusion is Conclusion.UNKNOWN
    assert result.verification_record_uri is None
    assert result.verification_record_uri is None
