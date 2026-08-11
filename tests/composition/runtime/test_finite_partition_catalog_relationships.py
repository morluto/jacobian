"""Finite-partition producer/checker discovery contract."""

from __future__ import annotations

from jacobian.contracts.capabilities import CapabilityCatalogRelationshipKind

# Composition-lane admission category for architecture ratchets.
COMPOSITION_ADMISSION = "AUTHORITY"


def test_finite_partition_producer_and_verifier_are_reciprocal(
    authorized_complete_runtime,
) -> None:
    catalog = {
        descriptor.capability_id: descriptor
        for descriptor in authorized_complete_runtime.core.capabilities.catalog().capabilities
    }
    producer_id = "case.partition.finite"
    verifier_id = "case.partition.finite.verify"
    assert catalog[verifier_id].provider_runtime.checker_ids != ()

    producer_links = {
        item.capability_id: item for item in catalog[producer_id].related_capabilities
    }
    verifier_links = {
        item.capability_id: item for item in catalog[verifier_id].related_capabilities
    }

    assert producer_links[verifier_id].kind is (
        CapabilityCatalogRelationshipKind.INDEPENDENT_VERIFIER
    )
    assert verifier_links[producer_id].kind is (
        CapabilityCatalogRelationshipKind.VERIFIABLE_RESULT_PRODUCER
    )
