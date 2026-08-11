"""Capability service composition and operator policy.

The service is deliberately a small composition root. Registration, discovery,
dispatch, verification, validation, and telemetry each live in their owning
module so the public service keeps one stable typed entry point without hiding
those boundaries behind a second compatibility API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from jacobian.canonical import canonicalize_json
from jacobian.capability_adapters import CapabilityAdapter
from jacobian.capability_discovery import (
    CapabilityDiscoveryMixin,
)
from jacobian.capability_discovery import (
    capability_domain as _capability_domain,
)
from jacobian.capability_discovery import (
    normalize_domain as _normalize_domain,
)
from jacobian.capability_dispatch import CapabilityDispatchMixin
from jacobian.capability_errors import (
    CapabilityDiscoveryCursorError,
    CapabilityError,
    CapabilityInvocationError,
)
from jacobian.capability_registry import CapabilityRegistryMixin
from jacobian.capability_verification import CapabilityVerificationMixin
from jacobian.contracts.capabilities import (
    CapabilityCatalogRelationship,
    CapabilityDescriptor,
)
from jacobian.storage.repository import ArtifactRepository


@dataclass(frozen=True, slots=True)
class CapabilityPolicy:
    """Operator-controlled visibility policy, separate from checker authority."""

    profile: Literal["DEFAULT", "COMPUTE_VERIFY_NO_RETRIEVAL"] = "DEFAULT"
    allowed_capability_ids: frozenset[str] = frozenset()
    denied_capability_ids: frozenset[str] = frozenset()
    allowed_domains: frozenset[str] = frozenset()
    denied_domains: frozenset[str] = frozenset()
    allowed_tags: frozenset[str] = frozenset()
    denied_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.profile not in {"DEFAULT", "COMPUTE_VERIFY_NO_RETRIEVAL"}:
            raise ValueError(f"unknown capability policy profile: {self.profile!r}")
        if self.profile == "COMPUTE_VERIFY_NO_RETRIEVAL":
            object.__setattr__(self, "denied_tags", self.denied_tags | {"retrieval"})
        for allowed, denied, label in (
            (self.allowed_capability_ids, self.denied_capability_ids, "capability IDs"),
            (self.allowed_domains, self.denied_domains, "domains"),
            (self.allowed_tags, self.denied_tags, "tags"),
        ):
            overlap = allowed & denied
            if overlap:
                raise ValueError(
                    f"capability policy allows and denies the same {label}: "
                    + ", ".join(sorted(str(item) for item in overlap))
                )
        for value in (
            *self.allowed_capability_ids,
            *self.denied_capability_ids,
            *self.allowed_domains,
            *self.denied_domains,
            *self.allowed_tags,
            *self.denied_tags,
        ):
            if not value.strip():
                raise ValueError("capability policy values must not be blank")

    @property
    def definition(self) -> dict[str, object]:
        return {
            "policy_version": "1",
            "profile": self.profile,
            "allowed_capability_ids": sorted(self.allowed_capability_ids),
            "denied_capability_ids": sorted(self.denied_capability_ids),
            "allowed_domains": sorted(self.allowed_domains),
            "denied_domains": sorted(self.denied_domains),
            "allowed_tags": sorted(self.allowed_tags),
            "denied_tags": sorted(self.denied_tags),
            "checker_authorization_affected": False,
        }

    @property
    def digest(self) -> str:
        import hashlib

        return (
            "sha256:" + hashlib.sha256(canonicalize_json(self.definition)).hexdigest()
        )

    def project(self, descriptor: CapabilityDescriptor) -> CapabilityDescriptor | None:
        if self.denial_reasons(descriptor):
            return None
        return descriptor

    def denial_reasons(
        self,
        descriptor: CapabilityDescriptor,
    ) -> tuple[str, ...]:
        capability_id = descriptor.capability_id
        domain = _normalize_domain(_capability_domain(descriptor))
        tags = {_normalize_domain(tag) for tag in descriptor.tags}
        reasons: list[str] = []
        if (
            self.allowed_capability_ids
            and capability_id not in self.allowed_capability_ids
        ):
            reasons.append("capability_id_not_allowed")
        if capability_id in self.denied_capability_ids:
            reasons.append("capability_id_denied")
        if self.allowed_domains and domain not in {
            _normalize_domain(value) for value in self.allowed_domains
        }:
            reasons.append("domain_not_allowed")
        if domain in {_normalize_domain(value) for value in self.denied_domains}:
            reasons.append("domain_denied")
        normalized_allowed_tags = {
            _normalize_domain(value) for value in self.allowed_tags
        }
        if normalized_allowed_tags and not tags & normalized_allowed_tags:
            reasons.append("tag_not_allowed")
        if tags & {_normalize_domain(value) for value in self.denied_tags}:
            reasons.append("tag_denied")
        return tuple(reasons)


class CapabilityService(
    CapabilityRegistryMixin,
    CapabilityDiscoveryMixin,
    CapabilityDispatchMixin,
    CapabilityVerificationMixin,
):
    """Typed composition root for capability lifecycle and invocation."""

    def __init__(
        self,
        store: ArtifactRepository,
        *,
        policy: CapabilityPolicy | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or CapabilityPolicy()
        self._adapters: dict[str, CapabilityAdapter] = {}
        self._descriptors: dict[str, CapabilityDescriptor] = {}
        self._catalog_relationships: dict[
            str, dict[str, CapabilityCatalogRelationship]
        ] = {}

    def _register_catalog_relationship(
        self,
        source_capability_id: str,
        relationship: CapabilityCatalogRelationship,
    ) -> None:
        """Register one installer-authorized directed catalog relationship."""

        if source_capability_id == relationship.capability_id:
            raise CapabilityError("a capability cannot relate to itself")
        related = self._catalog_relationships.setdefault(source_capability_id, {})
        previous = related.get(relationship.capability_id)
        if previous is not None and previous != relationship:
            raise CapabilityError(
                "conflicting catalog relationship: "
                f"{source_capability_id} -> {relationship.capability_id}"
            )
        related[relationship.capability_id] = relationship


__all__ = [
    "CapabilityAdapter",
    "CapabilityDiscoveryCursorError",
    "CapabilityError",
    "CapabilityInvocationError",
    "CapabilityPolicy",
    "CapabilityService",
]
