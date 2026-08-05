from __future__ import annotations

from pathlib import Path

from jacobian.artifacts import ArtifactService
from jacobian.claims import ClaimValidationService
from jacobian.contracts.claims import ClaimSpec
from jacobian.contracts.plugins import PluginManifest
from jacobian.plugins.registry import PluginRegistry
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository


def test_claim_validation_rejects_missing_plugin_capability(tmp_path: Path) -> None:
    store = ArtifactRepository(tmp_path)
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    claim_schema = schemas.register(
        name="jacobian.claim",
        version="1",
        schema=ClaimSpec.model_json_schema(),
    )
    candidate_schema = schemas.register(
        name="example.candidate",
        version="1",
        schema={"type": "object"},
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
        name="example.domain",
        version="1",
        definition={"description": "example finite domain"},
    )
    plugins = PluginRegistry(store)
    evaluator = plugins.register_implementation("jacobian_checkers.reject:check")
    manifest = artifacts.put(
        schema_uri=manifest_schema,
        semantics_uri=system_semantics,
        payload={
            "plugin_schema_version": "1",
            "domain_id": "example.domain",
            "domain_version": "1",
            "semantics_uri": domain_semantics,
            "claim_schema_uri": claim_schema,
            "candidate_schema_uri": candidate_schema,
            "witness_schema_uris": [],
            "certificate_schema_uris": [],
            "capabilities": {
                "Evaluator": {
                    "implementation_uri": evaluator,
                    "entrypoint": "jacobian_checkers.reject:check",
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
            "domain_id": "example.domain",
            "domain_version": "1",
            "semantics_uri": domain_semantics,
            "quantifiers": [],
            "predicate": {
                "name": "example_predicate",
                "parameters": {},
            },
            "bounds": {},
            "required_capabilities": ["WitnessOracle"],
            "correspondence_status": "UNREVIEWED",
        },
    )

    result = ClaimValidationService(store, schemas, plugins).validate(
        claim_uri=claim.artifact_uri,
        plugin_id=manifest.artifact_uri,
    )

    assert not result.valid
    assert result.input.status.value == "REJECTED"
    assert "WitnessOracle" in result.missing_capabilities
    assert "math.find" in result.input.errors[0]
