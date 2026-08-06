from __future__ import annotations

from pathlib import Path

import pytest

from jacobian.artifacts import ArtifactService
from jacobian.claims import ClaimValidationService
from jacobian.contracts.claims import ClaimSpec
from jacobian.contracts.evidence import WitnessEnvelope
from jacobian.contracts.plugins import PluginManifest
from jacobian.contracts.results import (
    Arithmetic,
    Assurance,
    Conclusion,
    Coverage,
    Execution,
    ExecutionStatus,
    InputStatus,
    InputValidation,
    Method,
    ResultEnvelope,
    Verification,
)
from jacobian.plugin_execution import PluginExecutionResult, PluginExecutor
from jacobian.plugins.registry import PluginRegistry
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService
from jacobian.witnesses import WitnessSearchService


def test_witness_search_stores_bound_unverified_evidence(tmp_path: Path) -> None:
    service, store, claim_uri, candidate_uri, plugin_id = _witness_fixture(tmp_path)

    response = service.find(
        claim_uri=claim_uri,
        candidate_uri=candidate_uri,
        plugin_id=plugin_id,
        witness_role="DEFEATS_CANDIDATE",
        wall_seconds=30,
    )

    assert response.status.value == "FOUND"
    assert response.result.assurance.verification.value == "UNVERIFIED"
    assert response.witness_uri is not None
    witness = WitnessEnvelope.model_validate(store.get(response.witness_uri).payload)
    assert witness.bindings.claim_digest == store.get(claim_uri).manifest.object_digest
    assert (
        witness.bindings.candidate_digest
        == store.get(candidate_uri).manifest.object_digest
    )


@pytest.mark.parametrize(
    ("certificate_conclusion", "expected_status"),
    [
        (Conclusion.TRUE, "NONE_CERTIFIED"),
        (Conclusion.FALSE, "UNKNOWN"),
    ],
)
def test_none_certified_requires_certificate_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    certificate_conclusion: Conclusion,
    expected_status: str,
) -> None:
    service, store, claim_uri, candidate_uri, plugin_id = _witness_fixture(tmp_path)
    certificate_uri = "artifact://sha256/" + "a" * 64
    claim = store.get(claim_uri)
    candidate = store.get(candidate_uri)
    semantics_digest = store.get(
        candidate.manifest.semantics_uri
    ).manifest.object_digest
    verified = ResultEnvelope(
        execution=Execution(status=ExecutionStatus.COMPLETED),
        input=InputValidation(status=InputStatus.ACCEPTED),
        conclusion=certificate_conclusion,
        assurance=Assurance(
            arithmetic=Arithmetic.EXACT_INTEGER,
            method=Method.EXHAUSTIVE_FINITE,
            coverage=Coverage.EXHAUSTIVE,
            verification=Verification.VERIFIED,
            checker_id="checker://sha256/" + "b" * 64,
            checker_digest="sha256:" + "c" * 64,
        ),
        claim_digest=claim.manifest.object_digest,
        semantics_digest=semantics_digest,
        candidate_digest=candidate.manifest.object_digest,
        evidence_uris=(certificate_uri,),
        verification_record_uri="artifact://sha256/" + "d" * 64,
    )
    monkeypatch.setattr(
        service.executor,
        "run",
        lambda **_kwargs: PluginExecutionResult(
            status=ExecutionStatus.COMPLETED,
            output={
                "status": "NONE_CERTIFIED",
                "certificate_uri": certificate_uri,
                "arithmetic": "EXACT_INTEGER",
                "coverage": "EXHAUSTIVE",
                "detail": "complete finite exhaustion",
            },
            diagnostics="",
            detail=None,
            runtime_ms=1,
        ),
    )
    monkeypatch.setattr(
        service.verification,
        "verify_certificate",
        lambda **_kwargs: verified,
    )

    response = service.find(
        claim_uri=claim_uri,
        candidate_uri=candidate_uri,
        plugin_id=plugin_id,
        witness_role="DEFEATS_CANDIDATE",
        wall_seconds=30,
    )

    assert response.status.value == expected_status
    if expected_status == "NONE_CERTIFIED":
        assert response.result.assurance.verification is Verification.VERIFIED
        assert response.certificate_uri == certificate_uri
    else:
        assert response.result.input.status is InputStatus.REJECTED
        assert response.certificate_uri is None


def _witness_fixture(
    root: Path,
) -> tuple[WitnessSearchService, ArtifactRepository, str, str, str]:
    store = ArtifactRepository(root)
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    claim_schema = schemas.register(
        name="jacobian.claim",
        version="1",
        schema=ClaimSpec.model_json_schema(),
    )
    candidate_schema = schemas.register(
        name="fixture.candidate",
        version="1",
        schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    manifest_schema = schemas.register(
        name="jacobian.plugin-manifest",
        version="1",
        schema=PluginManifest.model_json_schema(),
    )
    system_semantics = store.register_descriptor(
        kind="semantics",
        name="jacobian.plugin-manifest",
        version="1",
        definition={"description": "plugin metadata"},
    )
    domain_semantics = store.register_descriptor(
        kind="semantics",
        name="fixture.domain",
        version="1",
        definition={"description": "fixture integers"},
    )
    plugins = PluginRegistry(store)
    implementation = plugins.register_implementation(
        "tests.support.search_entrypoints:find_fixture_witness"
    )
    manifest = artifacts.put(
        schema_uri=manifest_schema,
        semantics_uri=system_semantics,
        payload={
            "plugin_schema_version": "1",
            "domain_id": "fixture.domain",
            "domain_version": "1",
            "semantics_uri": domain_semantics,
            "claim_schema_uri": claim_schema,
            "candidate_schema_uri": candidate_schema,
            "witness_schema_uris": [],
            "certificate_schema_uris": [],
            "capabilities": {
                "WitnessOracle": {
                    "implementation_uri": implementation,
                    "entrypoint": (
                        "tests.support.search_entrypoints:find_fixture_witness"
                    ),
                    "version": "1",
                }
            },
        },
    )
    plugins.install(manifest.artifact_uri)
    claim = artifacts.put(
        schema_uri=claim_schema,
        semantics_uri=domain_semantics,
        payload={
            "claim_schema_version": "1",
            "domain_id": "fixture.domain",
            "domain_version": "1",
            "semantics_uri": domain_semantics,
            "quantifiers": [],
            "predicate": {"name": "fixture_predicate", "parameters": {}},
            "bounds": {},
            "required_capabilities": ["WitnessOracle"],
            "correspondence_status": "UNREVIEWED",
        },
    )
    candidate = artifacts.put(
        schema_uri=candidate_schema,
        semantics_uri=domain_semantics,
        payload={"value": 3},
    )
    return (
        WitnessSearchService(
            store,
            schemas,
            plugins,
            ClaimValidationService(store, schemas, plugins),
            PluginExecutor(),
            VerificationService(store, CheckerRegistry(store)),
        ),
        store,
        claim.artifact_uri,
        candidate.artifact_uri,
        manifest.artifact_uri,
    )
