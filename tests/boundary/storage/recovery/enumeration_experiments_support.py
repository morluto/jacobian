"""Claim and plugin builders for enumeration experiment boundary tests."""

from __future__ import annotations

from jacobian.contracts.plugins import PluginManifest
from jacobian.runtime.model import JacobianRuntime


def _claim(
    runtime: JacobianRuntime,
    *,
    reference_name: str,
    predicate: str,
    parameters: dict[str, object],
) -> tuple[str, str]:
    reference = runtime.portfolio.references[reference_name]
    claim = runtime.core.artifacts.put(
        schema_uri=reference.claim_schema_uri,
        semantics_uri=reference.semantics_uri,
        payload={
            "claim_schema_version": "1",
            "domain_id": (
                "jacobian.graph-paths"
                if reference_name == "graph_paths"
                else "jacobian.integer-matrices"
            ),
            "domain_version": "1",
            "semantics_uri": reference.semantics_uri,
            "quantifiers": [],
            "predicate": {"name": predicate, "parameters": parameters},
            "bounds": {},
            "required_capabilities": ["CandidateEnumerator", "Evaluator"],
            "correspondence_status": "HUMAN_REVIEWED",
        },
    )
    return claim.artifact_uri, reference.plugin_id


def _install_matrix_enumerator_plugin(
    runtime: JacobianRuntime,
    *,
    entrypoint: str,
    evaluator_entrypoint: str = "jacobian.plugins.matrices:evaluate_capability",
) -> str:
    matrix = runtime.portfolio.references["matrices"]
    enumerator = runtime.core.plugins.register_implementation(entrypoint)
    evaluator = runtime.core.plugins.register_implementation(evaluator_entrypoint)
    manifest = runtime.core.artifacts.put(
        schema_uri=runtime.services.reference_installer.manifest_schema_uri,
        semantics_uri=runtime.services.reference_installer.manifest_semantics_uri,
        payload=PluginManifest(
            domain_id="jacobian.integer-matrices",
            domain_version="1",
            semantics_uri=matrix.semantics_uri,
            claim_schema_uri=matrix.claim_schema_uri,
            candidate_schema_uri=matrix.candidate_schema_uri,
            capabilities={
                "CandidateEnumerator": {
                    "implementation_uri": enumerator,
                    "entrypoint": entrypoint,
                    "version": "1",
                },
                "Evaluator": {
                    "implementation_uri": evaluator,
                    "entrypoint": evaluator_entrypoint,
                    "version": "1",
                },
            },
        ).model_dump(mode="json"),
    )
    runtime.core.plugins.install(manifest.artifact_uri)
    return manifest.artifact_uri


def _matrix_claim_for_plugin(
    runtime: JacobianRuntime,
    *,
    plugin_id: str,
) -> str:
    matrix = runtime.portfolio.references["matrices"]
    claim = runtime.core.artifacts.put(
        schema_uri=matrix.claim_schema_uri,
        semantics_uri=matrix.semantics_uri,
        payload={
            "claim_schema_version": "1",
            "domain_id": "jacobian.integer-matrices",
            "domain_version": "1",
            "semantics_uri": matrix.semantics_uri,
            "quantifiers": [],
            "predicate": {"name": "is_nonsingular", "parameters": {}},
            "bounds": {},
            "required_capabilities": ["CandidateEnumerator", "Evaluator"],
            "correspondence_status": "HUMAN_REVIEWED",
        },
    )
    validation = runtime.services.claims.validate(
        claim_uri=claim.artifact_uri,
        plugin_id=plugin_id,
    )
    assert validation.valid
    return claim.artifact_uri
