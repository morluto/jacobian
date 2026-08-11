from __future__ import annotations

from tests.support.capabilities import invoke_capability as _invoke

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityMode,
)
from jacobian.contracts.evidence import WitnessRole
from jacobian.contracts.results import Conclusion, Verification
from jacobian.references import ReferenceInstallation


def _claim(
    reference: ReferenceInstallation,
    *,
    capabilities: list[str],
) -> dict[str, object]:
    return {
        "claim_schema_version": "1",
        "domain_id": reference.domain_id,
        "domain_version": reference.domain_version,
        "semantics_uri": reference.semantics_uri,
        "quantifiers": [],
        "predicate": {"name": "is_bipartite", "parameters": {}},
        "bounds": {},
        "required_capabilities": capabilities,
        "correspondence_status": "HUMAN_REVIEWED",
    }


def test_atomic_capability_catalog_includes_required_and_excludes_composite_operations(
    attached_complete_runtime,
) -> None:
    runtime = attached_complete_runtime
    catalog = runtime.core.capabilities.catalog().capabilities
    ids = {item.capability_id for item in catalog}
    descriptors = {item.capability_id: item for item in catalog}

    assert {
        "artifact.put",
        "claim.validate",
        "evaluate.batch",
        "witness.find",
        "witness.verify",
        "certificate.verify",
        "shrink.run",
        "structure.canonicalize",
        "search.enumerate",
        "experiment.inspect",
        "experiment.wait",
        "experiment.cancel",
        "transform.apply",
        "transform.verify",
        "polytope.separate",
        "parameter.region.promote",
    }.issubset(ids)
    assert {
        "reference.solve",
        "verification.run",
        "search.run",
        "conjecture.generate",
        "conjecture.repair",
        "parameter.generalize",
    }.isdisjoint(ids)
    assert "witness_uri" in descriptors["witness.find"].output_schema["properties"]
    assert (
        "experiment_uri" in descriptors["search.enumerate"].output_schema["properties"]
    )


def test_atomic_capabilities_preserve_stage_assurance_and_checker_boundary(
    authorized_complete_runtime,
) -> None:
    reference = authorized_complete_runtime.portfolio.references["graph_paths"]

    claim = _invoke(
        authorized_complete_runtime,
        "artifact.put",
        {
            "schema_uri": reference.claim_schema_uri,
            "semantics_uri": reference.semantics_uri,
            "payload": _claim(reference, capabilities=["Evaluator", "WitnessOracle"]),
            "summary": "graph_paths claim",
        },
    )
    claim_uri = claim.output["artifact_uri"]
    candidate = _invoke(
        authorized_complete_runtime,
        "artifact.put",
        {
            "schema_uri": reference.candidate_schema_uri,
            "semantics_uri": reference.semantics_uri,
            "payload": {
                "vertices": ["a", "b", "c", "d"],
                "arcs": [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"]],
            },
            "parents": [claim_uri],
            "summary": "graph_paths candidate",
        },
    )
    candidate_uri = candidate.output["artifact_uri"]
    validation = _invoke(
        authorized_complete_runtime,
        "claim.validate",
        {"claim_uri": claim_uri, "plugin_id": reference.plugin_id},
    )
    evaluation = _invoke(
        authorized_complete_runtime,
        "evaluate.batch",
        {
            "claim_uri": claim_uri,
            "candidate_uris": [candidate_uri],
            "plugin_id": reference.plugin_id,
            "profile": "FAST",
            "seed": 0,
            "wall_seconds": 60,
        },
    )
    found = _invoke(
        authorized_complete_runtime,
        "witness.find",
        {
            "claim_uri": claim_uri,
            "candidate_uri": candidate_uri,
            "plugin_id": reference.plugin_id,
            "witness_role": WitnessRole.SUPPORTS_CLAIM.value,
            "wall_seconds": 300,
        },
    )
    witness_uri = found.output["witness_uri"]
    assert witness_uri is not None
    verified = _invoke(
        authorized_complete_runtime,
        "witness.verify",
        {
            "claim_uri": claim_uri,
            "candidate_uri": candidate_uri,
            "witness_uri": witness_uri,
            "checker_id": reference.witness_checker_ids["graph.2coloring"],
        },
        mode=CapabilityMode.VERIFY,
    )

    assert validation.output["valid"] is True
    assert validation.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert evaluation.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert found.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert all(
        item["result"]["assurance"]["verification"] == Verification.UNVERIFIED
        for item in evaluation.output["items"]
    )
    assert found.output["result"]["assurance"]["verification"] == (
        Verification.UNVERIFIED
    )
    assert verified.output["conclusion"] == Conclusion.TRUE
    assert verified.output["assurance"]["verification"] == Verification.VERIFIED
    assert verified.assurance.verification_record_uri is not None
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.completeness.status is CapabilityCompletenessStatus.NOT_APPLICABLE


def test_claim_validation_exposes_an_invalid_claim_without_composing_a_workflow(
    authorized_complete_runtime,
) -> None:
    reference = authorized_complete_runtime.portfolio.references["graph_paths"]

    claim = _invoke(
        authorized_complete_runtime,
        "artifact.put",
        {
            "schema_uri": reference.claim_schema_uri,
            "semantics_uri": reference.semantics_uri,
            "payload": _claim(reference, capabilities=["HypothesisTransformer"]),
        },
    )
    validation = _invoke(
        authorized_complete_runtime,
        "claim.validate",
        {"claim_uri": claim.output["artifact_uri"], "plugin_id": reference.plugin_id},
    )

    assert validation.output["valid"] is False
