"""Whole-portfolio discovery coverage for finite-poset intent."""

from __future__ import annotations

from jacobian.contracts.capabilities import (
    CapabilityDiscoveryRequest,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus

# Composition-lane admission category for architecture ratchets.
COMPOSITION_ADMISSION = "DISCOVERY"


def test_poset_width_intents_discover_the_domain_owned_operation(
    attached_complete_runtime_read_only,
) -> None:
    for query in (
        "maximum antichain and minimum chain decomposition of a finite poset",
        "compute the width of a finite partially ordered set",
    ):
        discovered = attached_complete_runtime_read_only.core.capabilities.discover(
            CapabilityDiscoveryRequest(query=query, limit=5)
        )

        assert discovered.matches[0].capability_id == "poset.width.compute"
        assert discovered.matches[0].lexical_fit == "STRONG_CANDIDATE"


def test_artifact_put_is_hidden_from_discovery_but_remains_dispatchable(
    attached_complete_runtime,
) -> None:
    catalog_ids = {
        descriptor.capability_id
        for descriptor in attached_complete_runtime.core.capabilities.catalog().capabilities
    }
    assert "artifact.put" in catalog_ids

    discovered = attached_complete_runtime.core.capabilities.discover(
        CapabilityDiscoveryRequest(query="store artifact", limit=20)
    )
    assert not any(
        match.capability_id == "artifact.put" for match in discovered.matches
    )

    result = attached_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="artifact.put",
            input={
                "schema_uri": "artifact://sha256/" + "0" * 64,
                "semantics_uri": "artifact://sha256/" + "0" * 64,
                "payload": {},
            },
        )
    )
    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code != "UNKNOWN_CAPABILITY"
