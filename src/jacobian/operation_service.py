"""Operation service composition and operator policy.

The service is deliberately a small composition root. Registration, discovery,
dispatch, verification, validation, and telemetry each live in their owning
module so the public service keeps one stable typed entry point without hiding
those boundaries behind a second compatibility API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from jacobian.canonical import canonicalize_json
from jacobian.contracts.operations import (
    OperationCatalogSnapshot,
    OperationDescriptor,
    OperationDiscoveryRequest,
    OperationDiscoveryResult,
    OperationRequest,
    OperationResult,
)
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_discovery import discover_operations
from jacobian.operation_discovery import (
    normalize_domain as _normalize_domain,
)
from jacobian.operation_discovery import (
    operation_domain as _operation_domain,
)
from jacobian.operation_dispatch import dispatch_operation
from jacobian.operation_errors import OperationError
from jacobian.operation_validation import validator
from jacobian.storage.repository import ArtifactRepository


@dataclass(frozen=True, slots=True)
class OperationVisibilityPolicy:
    """Operator-controlled visibility policy, separate from checker authority."""

    profile: Literal["DEFAULT", "COMPUTE_VERIFY_NO_RETRIEVAL"] = "DEFAULT"
    allowed_operation_ids: frozenset[str] = frozenset()
    denied_operation_ids: frozenset[str] = frozenset()
    allowed_domains: frozenset[str] = frozenset()
    denied_domains: frozenset[str] = frozenset()
    allowed_tags: frozenset[str] = frozenset()
    denied_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.profile not in {"DEFAULT", "COMPUTE_VERIFY_NO_RETRIEVAL"}:
            raise ValueError(f"unknown operation policy profile: {self.profile!r}")
        if self.profile == "COMPUTE_VERIFY_NO_RETRIEVAL":
            object.__setattr__(self, "denied_tags", self.denied_tags | {"retrieval"})
        for allowed, denied, label in (
            (self.allowed_operation_ids, self.denied_operation_ids, "operation IDs"),
            (self.allowed_domains, self.denied_domains, "domains"),
            (self.allowed_tags, self.denied_tags, "tags"),
        ):
            overlap = allowed & denied
            if overlap:
                raise ValueError(
                    f"operation policy allows and denies the same {label}: "
                    + ", ".join(sorted(str(item) for item in overlap))
                )
        for value in (
            *self.allowed_operation_ids,
            *self.denied_operation_ids,
            *self.allowed_domains,
            *self.denied_domains,
            *self.allowed_tags,
            *self.denied_tags,
        ):
            if not value.strip():
                raise ValueError("operation policy values must not be blank")

    @property
    def definition(self) -> dict[str, object]:
        return {
            "policy_version": "1",
            "profile": self.profile,
            "allowed_operation_ids": sorted(self.allowed_operation_ids),
            "denied_operation_ids": sorted(self.denied_operation_ids),
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

    def project(self, descriptor: OperationDescriptor) -> OperationDescriptor | None:
        if self.denial_reasons(descriptor):
            return None
        return descriptor

    def denial_reasons(
        self,
        descriptor: OperationDescriptor,
    ) -> tuple[str, ...]:
        operation_id = descriptor.operation_id
        domain = _normalize_domain(_operation_domain(descriptor))
        tags = {_normalize_domain(tag) for tag in descriptor.tags}
        reasons: list[str] = []
        if (
            self.allowed_operation_ids
            and operation_id not in self.allowed_operation_ids
        ):
            reasons.append("operation_id_not_allowed")
        if operation_id in self.denied_operation_ids:
            reasons.append("operation_id_denied")
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


class OperationService:
    """Typed composition root for operation lifecycle and invocation."""

    def __init__(
        self,
        store: ArtifactRepository,
        *,
        policy: OperationVisibilityPolicy | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or OperationVisibilityPolicy()
        self._adapters: dict[str, OperationAdapter[Any]] = {}
        self._descriptors: dict[str, OperationDescriptor] = {}

    def register(self, adapter: OperationAdapter[Any]) -> None:
        descriptor = adapter.descriptor
        if descriptor.operation_id in self._adapters:
            raise OperationError(f"duplicate operation ID: {descriptor.operation_id}")
        validator(descriptor.input_schema)
        validator(descriptor.output_schema)
        for example in descriptor.examples:
            try:
                from jacobian.operation_validation import validate_payload

                validate_payload(descriptor.input_schema, example.input)
            except OperationError as exc:
                raise OperationError(
                    f"operation {descriptor.operation_id} invocation example "
                    f"{example.name!r} does not match its input schema"
                ) from exc
        self._descriptors[descriptor.operation_id] = descriptor.model_copy(deep=True)
        self._adapters[descriptor.operation_id] = adapter

    def catalog(self) -> OperationCatalogSnapshot:
        projected = tuple(
            projected
            for name in sorted(self._adapters)
            if (projected := self.policy.project(self._descriptors[name])) is not None
        )
        return OperationCatalogSnapshot(
            policy_profile=self.policy.profile,
            policy_digest=self.policy.digest,
            operations=projected,
        )

    def inspect(self, operation_id: str) -> OperationDescriptor | None:
        descriptor = self._descriptors.get(operation_id)
        if descriptor is None:
            return None
        return self.policy.project(descriptor)

    def discover(self, request: OperationDiscoveryRequest) -> OperationDiscoveryResult:
        return discover_operations(self.catalog(), request)

    def invoke(self, request: OperationRequest) -> OperationResult:
        return dispatch_operation(self, request)

    def _validate_verified_result(self, result: Any) -> None:
        from jacobian.operation_verification import validate_verified_result

        validate_verified_result(self.store, result)


__all__ = [
    "OperationService",
    "OperationVisibilityPolicy",
]
