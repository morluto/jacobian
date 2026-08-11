from __future__ import annotations

from tests.support.capabilities import invoke_capability as _invoke
from tests.support.services import atomic_installation, open_reference_services

from jacobian.atomic_capabilities import install_atomic_capabilities
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
)
from jacobian.contracts.evidence import WitnessRole
from jacobian.contracts.results import Conclusion, Verification
from jacobian.references import ReferenceInstallation

# Composition-lane admission category for architecture ratchets.
COMPOSITION_ADMISSION = "AUTHORITY"


def _claim(reference: ReferenceInstallation) -> dict[str, object]:
    return {
        "claim_schema_version": "1",
        "domain_id": reference.domain_id,
        "domain_version": reference.domain_version,
        "semantics_uri": reference.semantics_uri,
        "quantifiers": [],
        "predicate": {"name": "is_bipartite", "parameters": {}},
        "bounds": {},
        "required_capabilities": ["Evaluator", "WitnessOracle"],
        "correspondence_status": "HUMAN_REVIEWED",
    }


def test_atomic_capabilities_preserve_stage_assurance_and_checker_boundary(
    tmp_path,
) -> None:
    with open_reference_services(
        tmp_path / "state", "graph_paths", authorize_checkers=True
    ) as services:
        with atomic_installation(services.core):
            for adapter in install_atomic_capabilities(
                services.installation,
                services.application,
            ):
                services.installation.register_capability(adapter)
        reference = services.references["graph_paths"]

        claim = _invoke(
            services,
            "artifact.put",
            {
                "schema_uri": reference.claim_schema_uri,
                "semantics_uri": reference.semantics_uri,
                "payload": _claim(reference),
                "summary": "graph_paths claim",
            },
        )
        claim_uri = claim.output["artifact_uri"]
        candidate = _invoke(
            services,
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
            services,
            "claim.validate",
            {"claim_uri": claim_uri, "plugin_id": reference.plugin_id},
        )
        evaluation = _invoke(
            services,
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
            services,
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
            services,
            "witness.verify",
            {
                "claim_uri": claim_uri,
                "candidate_uri": candidate_uri,
                "witness_uri": witness_uri,
                "checker_id": reference.witness_checker_ids["graph.2coloring"],
            },
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
        assert (
            verified.completeness.status is CapabilityCompletenessStatus.NOT_APPLICABLE
        )
