"""Catalog relationship propagation and policy-boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.composition.runtime.capability_service_support import (
    ComputedAdapter,
    DiscoveryAdapter,
)

from jacobian.capability_service import (
    CapabilityError,
    CapabilityPolicy,
    CapabilityService,
)
from jacobian.contracts.capabilities import (
    CapabilityCatalogRelationship,
    CapabilityCatalogRelationshipKind,
    CapabilityProviderAvailability,
)
from jacobian.domains.builtins import build_builtin_domain_bundles
from jacobian.storage.repository import ArtifactRepository


def _relationship(target: str) -> CapabilityCatalogRelationship:
    return CapabilityCatalogRelationship(
        capability_id=target,
        kind=CapabilityCatalogRelationshipKind.INDEPENDENT_VERIFIER,
        relationship="independently verify this exact producer result",
    )


def test_all_authorized_exact_checker_pairs_are_reciprocal(
    authorized_complete_runtime,
) -> None:
    catalog = {
        descriptor.capability_id: descriptor
        for descriptor in authorized_complete_runtime.core.capabilities.catalog().capabilities
    }
    authoritative_pairs = {
        (declaration.capability_id, declaration.verification_capability_id)
        for bundle in build_builtin_domain_bundles()
        for declaration in bundle.checker_declarations
        if declaration.verification_capability_id is not None
        and declaration.capability_id in catalog
        and declaration.verification_capability_id in catalog
    }

    assert len(authoritative_pairs) == 56
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


def test_catalog_relationship_hides_missing_and_policy_hidden_endpoints(
    tmp_path: Path,
) -> None:
    source_id = ComputedAdapter.descriptor.capability_id
    target_id = "example.hidden-target"
    store = ArtifactRepository(tmp_path / "state")
    try:
        service = CapabilityService(
            store,
            policy=CapabilityPolicy(denied_capability_ids=frozenset({target_id})),
        )
        service.register(ComputedAdapter())
        service.register(
            DiscoveryAdapter(
                ComputedAdapter.descriptor.model_copy(
                    update={"capability_id": target_id, "title": "Hidden target"}
                )
            )
        )
        service.register_catalog_relationship(source_id, _relationship(target_id))

        catalog = {
            descriptor.capability_id: descriptor
            for descriptor in service.catalog().capabilities
        }
        assert target_id not in catalog
        assert catalog[source_id].related_capabilities == ()

        missing_id = "example.missing-target"
        service.register_catalog_relationship(source_id, _relationship(missing_id))
        catalog = {
            descriptor.capability_id: descriptor
            for descriptor in service.catalog().capabilities
        }
        assert catalog[source_id].related_capabilities == ()
    finally:
        store.close()


def test_catalog_relationship_hides_unavailable_endpoint_and_rejects_bad_edges(
    tmp_path: Path,
) -> None:
    source_id = ComputedAdapter.descriptor.capability_id
    target_id = "example.unavailable-target"
    store = ArtifactRepository(tmp_path / "state")
    try:
        service = CapabilityService(store)
        service.register(ComputedAdapter())
        unavailable_runtime = ComputedAdapter.descriptor.provider_runtime.model_copy(
            update={
                "availability": CapabilityProviderAvailability.UNAVAILABLE,
                "version": None,
                "digest": None,
                "digest_kind": None,
                "diagnostic": "fixture unavailable",
            }
        )
        unavailable = DiscoveryAdapter(
            ComputedAdapter.descriptor.model_copy(
                update={
                    "capability_id": target_id,
                    "title": "Unavailable target",
                    "provider_runtime": unavailable_runtime,
                }
            )
        )
        with pytest.raises(CapabilityError, match="is unavailable"):
            service.register(unavailable)
        service.register_catalog_relationship(source_id, _relationship(target_id))
        assert service.catalog().capabilities[0].related_capabilities == ()

        with pytest.raises(CapabilityError, match="cannot relate to itself"):
            service.register_catalog_relationship(source_id, _relationship(source_id))
        service.register_catalog_relationship(source_id, _relationship("example.other"))
        with pytest.raises(CapabilityError, match="conflicting catalog relationship"):
            service.register_catalog_relationship(
                source_id,
                CapabilityCatalogRelationship(
                    capability_id="example.other",
                    kind=(CapabilityCatalogRelationshipKind.VERIFIABLE_RESULT_PRODUCER),
                    relationship="conflicting fixture relationship",
                ),
            )
    finally:
        store.close()


@pytest.mark.parametrize(
    "kind",
    [
        CapabilityCatalogRelationshipKind.INDEPENDENT_VERIFIER,
        CapabilityCatalogRelationshipKind.VERIFIABLE_RESULT_PRODUCER,
    ],
)
def test_adapter_descriptors_cannot_authorize_verification_relationships(
    tmp_path: Path,
    kind: CapabilityCatalogRelationshipKind,
) -> None:
    target_id = "example.ordinary-computation"
    relationship = CapabilityCatalogRelationship(
        capability_id=target_id,
        kind=kind,
        relationship="untrusted verification-sensitive navigation",
    )
    source = DiscoveryAdapter(
        ComputedAdapter.descriptor.model_copy(
            update={"related_capabilities": (relationship,)}
        )
    )
    store = ArtifactRepository(tmp_path / "state")
    try:
        service = CapabilityService(store)
        service.register(
            DiscoveryAdapter(
                ComputedAdapter.descriptor.model_copy(
                    update={
                        "capability_id": target_id,
                        "title": "Ordinary computation",
                    }
                )
            )
        )

        with pytest.raises(
            CapabilityError,
            match="operator-authorized checker registration",
        ):
            service.register(source)
    finally:
        store.close()
