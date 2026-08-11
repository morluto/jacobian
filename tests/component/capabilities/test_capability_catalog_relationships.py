"""Catalog relationship propagation and policy-boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.component.capabilities.capability_service_support import (
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
from jacobian.storage.repository import ArtifactRepository


def _relationship(target: str) -> CapabilityCatalogRelationship:
    return CapabilityCatalogRelationship(
        capability_id=target,
        kind=CapabilityCatalogRelationshipKind.INDEPENDENT_VERIFIER,
        relationship="independently verify this exact producer result",
    )


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
        service._register_catalog_relationship(source_id, _relationship(target_id))

        catalog = {
            descriptor.capability_id: descriptor
            for descriptor in service.catalog().capabilities
        }
        assert target_id not in catalog
        assert catalog[source_id].related_capabilities == ()

        missing_id = "example.missing-target"
        service._register_catalog_relationship(source_id, _relationship(missing_id))
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
        service._register_catalog_relationship(source_id, _relationship(target_id))
        assert service.catalog().capabilities[0].related_capabilities == ()

        with pytest.raises(CapabilityError, match="cannot relate to itself"):
            service._register_catalog_relationship(source_id, _relationship(source_id))
        service._register_catalog_relationship(
            source_id, _relationship("example.other")
        )
        with pytest.raises(CapabilityError, match="conflicting catalog relationship"):
            service._register_catalog_relationship(
                source_id,
                CapabilityCatalogRelationship(
                    capability_id="example.other",
                    kind=(CapabilityCatalogRelationshipKind.VERIFIABLE_RESULT_PRODUCER),
                    relationship="conflicting fixture relationship",
                ),
            )
    finally:
        store.close()


def test_capability_service_has_no_public_relationship_authority(
    tmp_path: Path,
) -> None:
    store = ArtifactRepository(tmp_path / "state")
    try:
        service = CapabilityService(store)
        assert not hasattr(service, "register_catalog_relationship")
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


def test_registration_snapshots_the_validated_descriptor(tmp_path: Path) -> None:
    target_id = "example.ordinary-computation"
    relationship = CapabilityCatalogRelationship(
        capability_id=target_id,
        kind=CapabilityCatalogRelationshipKind.INDEPENDENT_VERIFIER,
        relationship="untrusted late relationship",
    )
    safe_descriptor = ComputedAdapter.descriptor
    unsafe_descriptor = safe_descriptor.model_copy(
        update={"related_capabilities": (relationship,)}
    )

    class StatefulDescriptorAdapter:
        descriptor_reads = 0

        @property
        def descriptor(self):
            self.descriptor_reads += 1
            return safe_descriptor if self.descriptor_reads == 1 else unsafe_descriptor

        def invoke(self, request):
            return ComputedAdapter().invoke(request)

    adapter = StatefulDescriptorAdapter()
    store = ArtifactRepository(tmp_path / "state")
    try:
        service = CapabilityService(store)
        service.register(adapter)

        descriptor = next(
            item
            for item in service.catalog().capabilities
            if item.capability_id == safe_descriptor.capability_id
        )

        assert adapter.descriptor_reads == 1
        assert descriptor.related_capabilities == ()
    finally:
        store.close()
