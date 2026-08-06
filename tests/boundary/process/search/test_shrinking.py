from __future__ import annotations

from pathlib import Path

from jacobian.artifacts import ArtifactService
from jacobian.claims import ClaimValidationService
from jacobian.contracts.claims import ClaimSpec
from jacobian.contracts.plugins import PluginManifest
from jacobian.plugin_execution import PluginExecutor
from jacobian.plugins.registry import PluginRegistry
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry
from jacobian.shrinking import ShrinkService
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService


def test_shrinker_rejects_nonpreserving_step_without_trusting_reducer_completeness(
    tmp_path: Path,
) -> None:
    (
        service,
        store,
        claim_uri,
        candidate_uri,
        plugin_id,
        checker_id,
    ) = _shrink_fixture(tmp_path)

    result = service.run(
        target_kind="candidate",
        target_uri=candidate_uri,
        claim_uri=claim_uri,
        plugin_id=plugin_id,
        preservation_checker_id=checker_id,
        reducers=("decrement",),
        objectives=("value",),
        evaluation_budget=10,
    )

    assert result.final_target_uri != candidate_uri
    assert store.get(result.final_target_uri).payload == {"value": 1}
    assert result.minimality.value == "LOCAL"
    assert any(not step.accepted for step in result.steps)
    assert result.result.execution.status.value == "COMPLETED"
    assert result.result.assurance.verification.value == "VERIFIED"


def test_shrinker_rejects_non_improving_proposal(tmp_path: Path) -> None:
    (
        service,
        _,
        claim_uri,
        candidate_uri,
        plugin_id,
        checker_id,
    ) = _shrink_fixture(
        tmp_path,
        reducer_entrypoint=(
            "tests.support.shrinking_entrypoints:reduce_without_improvement"
        ),
    )

    result = service.run(
        target_kind="candidate",
        target_uri=candidate_uri,
        claim_uri=claim_uri,
        plugin_id=plugin_id,
        preservation_checker_id=checker_id,
        reducers=("decrement",),
        objectives=("value",),
        evaluation_budget=3,
    )

    assert result.final_target_uri == candidate_uri
    assert result.minimality.value == "NONE"
    assert result.steps[0].accepted is False
    assert "strictly improve" in result.steps[0].detail


def test_shrinker_does_not_trust_empty_reducer_response_for_minimality(
    tmp_path: Path,
) -> None:
    (
        service,
        store,
        claim_uri,
        candidate_uri,
        plugin_id,
        checker_id,
    ) = _shrink_fixture(
        tmp_path,
        reducer_entrypoint=(
            "tests.support.shrinking_entrypoints:reduce_once_then_claim_complete"
        ),
    )

    result = service.run(
        target_kind="candidate",
        target_uri=candidate_uri,
        claim_uri=claim_uri,
        plugin_id=plugin_id,
        preservation_checker_id=checker_id,
        reducers=("decrement",),
        objectives=("value",),
        evaluation_budget=3,
    )

    assert store.get(result.final_target_uri).payload == {"value": 2}
    assert result.result.execution.status.value == "COMPLETED"
    assert result.result.assurance.verification.value == "VERIFIED"
    assert result.minimality.value == "NONE"


def test_shrinker_does_not_treat_checker_error_as_boundary_rejection(
    tmp_path: Path,
) -> None:
    (
        service,
        store,
        claim_uri,
        candidate_uri,
        plugin_id,
        checker_id,
    ) = _shrink_fixture(
        tmp_path,
        checker_entrypoint=(
            "tests.support.shrinking_entrypoints:preserve_positive_except_failed_boundary"
        ),
    )

    result = service.run(
        target_kind="candidate",
        target_uri=candidate_uri,
        claim_uri=claim_uri,
        plugin_id=plugin_id,
        preservation_checker_id=checker_id,
        reducers=("decrement",),
        objectives=("value",),
        evaluation_budget=4,
    )

    assert store.get(result.final_target_uri).payload == {"value": 2}
    assert result.result.execution.status.value == "COMPLETED"
    assert result.result.assurance.verification.value == "VERIFIED"
    assert result.minimality.value == "NONE"
    assert result.steps[-1].accepted is False


def _shrink_fixture(
    root: Path,
    *,
    reducer_entrypoint: str = (
        "tests.support.shrinking_entrypoints:reduce_positive_value"
    ),
    checker_entrypoint: str = ("tests.support.shrinking_entrypoints:preserve_positive"),
) -> tuple[ShrinkService, ArtifactRepository, str, str, str, str]:
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
    manifest_semantics = store.register_descriptor(
        kind="semantics",
        name="jacobian.plugin-manifest",
        version="1",
        definition={"description": "plugin metadata"},
    )
    semantics = store.register_descriptor(
        kind="semantics",
        name="fixture.positive",
        version="1",
        definition={"description": "positive integer counterexample fixture"},
    )
    plugins = PluginRegistry(store)
    implementation = plugins.register_implementation(reducer_entrypoint)
    manifest = artifacts.put(
        schema_uri=manifest_schema,
        semantics_uri=manifest_semantics,
        payload={
            "plugin_schema_version": "1",
            "domain_id": "fixture.positive",
            "domain_version": "1",
            "semantics_uri": semantics,
            "claim_schema_uri": claim_schema,
            "candidate_schema_uri": candidate_schema,
            "witness_schema_uris": [],
            "certificate_schema_uris": [],
            "capabilities": {
                "Reducer": {
                    "implementation_uri": implementation,
                    "entrypoint": reducer_entrypoint,
                    "version": "1",
                }
            },
        },
    )
    plugins.install(manifest.artifact_uri)
    claim = artifacts.put(
        schema_uri=claim_schema,
        semantics_uri=semantics,
        payload={
            "claim_schema_version": "1",
            "domain_id": "fixture.positive",
            "domain_version": "1",
            "semantics_uri": semantics,
            "quantifiers": [],
            "predicate": {"name": "is_not_positive", "parameters": {}},
            "bounds": {},
            "required_capabilities": ["Reducer"],
            "correspondence_status": "UNREVIEWED",
        },
    )
    candidate = artifacts.put(
        schema_uri=candidate_schema,
        semantics_uri=semantics,
        payload={"value": 3},
    )
    checkers = CheckerRegistry(store)
    checker = checkers.authorize(
        name="positive-preservation fixture",
        entrypoint=checker_entrypoint,
        evidence_kind="PRESERVATION",
        format_id="fixture.positive",
        format_version="1",
        claim_schema_uris=(claim_schema,),
        semantics_uris=(semantics,),
        candidate_schema_uris=(candidate_schema,),
    )
    verification = VerificationService(store, checkers)
    return (
        ShrinkService(
            store,
            schemas,
            plugins,
            ClaimValidationService(store, schemas, plugins),
            PluginExecutor(),
            verification,
        ),
        store,
        claim.artifact_uri,
        candidate.artifact_uri,
        manifest.artifact_uri,
        checker.checker_id,
    )
