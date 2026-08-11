"""Whole-portfolio visibility and dispatch contract for artifact.put."""

from __future__ import annotations

from jacobian.contracts.capabilities import (
    CapabilityDiscoveryRequest,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus

# Composition-lane admission category for architecture ratchets.
COMPOSITION_ADMISSION = "DISCOVERY"


def test_artifact_put_is_hidden_from_discovery_but_remains_dispatchable(
    attached_complete_runtime,
) -> None:
    capabilities = attached_complete_runtime.core.capabilities
    catalog_ids = {
        descriptor.capability_id for descriptor in capabilities.catalog().capabilities
    }
    assert "artifact.put" in catalog_ids

    discovered = capabilities.discover(
        CapabilityDiscoveryRequest(query="store artifact", limit=20)
    )
    assert not any(
        match.capability_id == "artifact.put" for match in discovered.matches
    )

    result = capabilities.invoke(
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
