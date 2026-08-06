from __future__ import annotations

from pathlib import Path

from jacobian.artifacts import ArtifactService
from jacobian.claims import ClaimValidationService
from jacobian.contracts.claims import ClaimSpec
from jacobian.contracts.plugins import PluginManifest
from jacobian.evaluation import EvaluationService
from jacobian.plugin_execution import PluginExecutor
from jacobian.plugins.registry import PluginRegistry
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository


def test_exact_exhaustive_evaluation_remains_unverified(tmp_path: Path) -> None:
    service, claim_uri, candidate_uri, plugin_id = _evaluation_fixture(tmp_path)

    response = service.evaluate_batch(
        claim_uri=claim_uri,
        candidate_uris=(candidate_uri,),
        plugin_id=plugin_id,
        profile="EXACT_CANDIDATE",
        seed=7,
        wall_seconds=30,
    )

    assert response.input.status.value == "ACCEPTED"
    assert response.evaluator_digest is not None
    assert response.environment_digest is not None
    item = response.items[0]
    assert item.result.conclusion.value == "FALSE"
    assert item.result.assurance.arithmetic.value == "EXACT_INTEGER"
    assert item.result.assurance.coverage.value == "EXHAUSTIVE"
    assert item.result.assurance.verification.value == "UNVERIFIED"
    assert item.objectives == {"violations": "1"}


def _evaluation_fixture(
    root: Path,
) -> tuple[EvaluationService, str, str, str]:
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
        "tests.support.search_entrypoints:evaluate_candidate"
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
                "Evaluator": {
                    "implementation_uri": implementation,
                    "entrypoint": (
                        "tests.support.search_entrypoints:evaluate_candidate"
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
            "required_capabilities": ["Evaluator"],
            "correspondence_status": "UNREVIEWED",
        },
    )
    candidate = artifacts.put(
        schema_uri=candidate_schema,
        semantics_uri=domain_semantics,
        payload={"value": 3},
    )
    return (
        EvaluationService(
            store,
            schemas,
            plugins,
            ClaimValidationService(store, schemas, plugins),
            PluginExecutor(),
        ),
        claim.artifact_uri,
        candidate.artifact_uri,
        manifest.artifact_uri,
    )
