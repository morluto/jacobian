"""Whole-portfolio discovery coverage for topology intent."""

from __future__ import annotations

from jacobian.contracts.capabilities import CapabilityDiscoveryRequest

# Composition-lane admission category for architecture ratchets.
COMPOSITION_ADMISSION = "DISCOVERY"


def test_homology_intent_discovers_the_domain_owned_operation(
    attached_complete_runtime_read_only,
) -> None:
    discovered = attached_complete_runtime_read_only.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="homology of a finite simplicial complex over F_2",
            limit=5,
        )
    )

    assert discovered.matches[0].capability_id == (
        "topology.simplicial_homology.compute"
    )
    assert discovered.matches[0].lexical_fit == "STRONG_CANDIDATE"
