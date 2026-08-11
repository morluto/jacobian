"""Whole-portfolio exact-checker catalog relationship contract."""

from __future__ import annotations

from jacobian.contracts.capabilities import CapabilityCatalogRelationshipKind
from jacobian.domains.builtins import build_builtin_domain_bundles

# Composition-lane admission category for architecture ratchets.
COMPOSITION_ADMISSION = "AUTHORITY"


def test_all_authorized_exact_checker_pairs_are_reciprocal(
    authorized_complete_runtime,
) -> None:
    catalog = {
        descriptor.capability_id: descriptor
        for descriptor in authorized_complete_runtime.core.capabilities.catalog().capabilities
    }
    assert any(
        descriptor.provider_runtime.checker_ids != () for descriptor in catalog.values()
    )
    authoritative_pairs = {
        (declaration.capability_id, declaration.verification_capability_id)
        for bundle in build_builtin_domain_bundles()
        for declaration in bundle.checker_declarations
        if declaration.verification_capability_id is not None
        and declaration.capability_id in catalog
        and declaration.verification_capability_id in catalog
    }

    assert authoritative_pairs
    for producer_id, verifier_id in authoritative_pairs:
        producer_links = {
            item.capability_id: item
            for item in catalog[producer_id].related_capabilities
        }
        verifier_links = {
            item.capability_id: item
            for item in catalog[verifier_id].related_capabilities
        }
        assert producer_links[verifier_id].kind is (
            CapabilityCatalogRelationshipKind.INDEPENDENT_VERIFIER
        )
        assert verifier_links[producer_id].kind is (
            CapabilityCatalogRelationshipKind.VERIFIABLE_RESULT_PRODUCER
        )

    modular_producer = "modular.polynomial_residue_image.compute"
    modular_verifier = "modular.polynomial_residue_image.verify"
    assert modular_verifier in {
        item.capability_id for item in catalog[modular_producer].related_capabilities
    }
    assert modular_producer in {
        item.capability_id for item in catalog[modular_verifier].related_capabilities
    }
