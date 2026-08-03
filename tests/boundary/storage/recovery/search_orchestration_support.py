"""Shared helpers for search orchestration recovery boundary tests."""

from __future__ import annotations

from jacobian.contracts.claims import ClaimSpec
from jacobian.contracts.evidence import WitnessRole
from jacobian.contracts.plugins import PluginManifest
from jacobian.contracts.search import SearchBudget, SearchRunRequest
from jacobian.runtime.model import JacobianRuntime


def _install_search_plugin(
    runtime: JacobianRuntime,
    *,
    proposer_entrypoint: str = (
        "tests.component.plugins._fixture_plugins:propose_fixture_values"
    ),
    refiner_entrypoint: str = (
        "tests.component.plugins._fixture_plugins:refine_fixture_search"
    ),
    include_witness_oracle: bool = False,
) -> tuple[str, str]:
    claim_schema_uri = runtime.core.schemas.register(
        name="fixture.search-claim",
        version="1",
        schema=ClaimSpec.model_json_schema(),
    )
    candidate_schema_uri = runtime.core.schemas.register(
        name="fixture.search-candidate",
        version="1",
        schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    semantics_uri = runtime.core.store.register_descriptor(
        kind="semantics",
        name="fixture.search-domain",
        version="1",
        definition={"description": "finite integer search fixture"},
    )
    entrypoints = {
        "Proposer": proposer_entrypoint,
        "Refiner": refiner_entrypoint,
        "Evaluator": "tests.component.plugins._fixture_plugins:evaluate_candidate",
    }
    if include_witness_oracle:
        entrypoints["WitnessOracle"] = (
            "tests.component.plugins._fixture_plugins:find_fixture_witness"
        )
    capabilities: dict[str, dict[str, str]] = {}
    for name, entrypoint in entrypoints.items():
        capabilities[name] = {
            "implementation_uri": runtime.core.plugins.register_implementation(
                entrypoint
            ),
            "entrypoint": entrypoint,
            "version": "1",
        }
    manifest = runtime.core.artifacts.put(
        schema_uri=runtime.services.reference_installer.manifest_schema_uri,
        semantics_uri=runtime.services.reference_installer.manifest_semantics_uri,
        payload=PluginManifest(
            domain_id="fixture.search-domain",
            domain_version="1",
            semantics_uri=semantics_uri,
            claim_schema_uri=claim_schema_uri,
            candidate_schema_uri=candidate_schema_uri,
            capabilities=capabilities,
        ).model_dump(mode="json"),
    )
    runtime.core.plugins.install(manifest.artifact_uri)
    claim = runtime.core.artifacts.put(
        schema_uri=claim_schema_uri,
        semantics_uri=semantics_uri,
        payload={
            "claim_schema_version": "1",
            "domain_id": "fixture.search-domain",
            "domain_version": "1",
            "semantics_uri": semantics_uri,
            "quantifiers": [],
            "predicate": {"name": "fixture_predicate", "parameters": {}},
            "bounds": {},
            "required_capabilities": ["Proposer", "Refiner", "Evaluator"],
            "correspondence_status": "UNREVIEWED",
        },
    )
    return claim.artifact_uri, manifest.artifact_uri


def _request(
    claim_uri: str,
    plugin_id: str,
    *,
    idempotency_key: str,
    batch_size: int = 1,
    wall_seconds: int = 30,
    witness_role: WitnessRole | None = None,
    counterexample_checker_id: str | None = None,
) -> SearchRunRequest:
    return SearchRunRequest(
        idempotency_key=idempotency_key,
        claim_uri=claim_uri,
        plugin_id=plugin_id,
        initial_state={"cursor": 0},
        witness_role=witness_role,
        counterexample_checker_id=counterexample_checker_id,
        budget=SearchBudget(
            candidates_max=8,
            iterations_max=8,
            wall_seconds=wall_seconds,
            batch_size=batch_size,
            workers=1,
        ),
    )
