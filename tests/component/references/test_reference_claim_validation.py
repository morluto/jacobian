from __future__ import annotations

from tests.support.capabilities import invoke_capability as _invoke
from tests.support.services import atomic_installation, open_reference_services

from jacobian.atomic_capabilities import install_atomic_capabilities


def test_claim_validation_exposes_an_invalid_claim(tmp_path) -> None:
    with open_reference_services(tmp_path / "state", "graph_paths") as services:
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
                "payload": {
                    "claim_schema_version": "1",
                    "domain_id": reference.domain_id,
                    "domain_version": reference.domain_version,
                    "semantics_uri": reference.semantics_uri,
                    "quantifiers": [],
                    "predicate": {"name": "is_bipartite", "parameters": {}},
                    "bounds": {},
                    "required_capabilities": ["HypothesisTransformer"],
                    "correspondence_status": "HUMAN_REVIEWED",
                },
            },
        )
        validation = _invoke(
            services,
            "claim.validate",
            {
                "claim_uri": claim.output["artifact_uri"],
                "plugin_id": reference.plugin_id,
            },
        )

        assert validation.output["valid"] is False
