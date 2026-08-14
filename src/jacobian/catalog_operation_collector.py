"""Operator-only collection of bound operations during catalog compilation."""

from __future__ import annotations

from typing import Any

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
from jacobian.operation_dispatch import dispatch_operation
from jacobian.operation_registration import register_operation
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.storage.repository import ArtifactRepository


class CatalogOperationCollector:
    """Collect descriptors and adapters only while compiling a catalog."""

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
        register_operation(adapter, self._adapters, self._descriptors)

    def snapshot(self) -> OperationCatalogSnapshot:
        projected = tuple(
            projected
            for operation_id in sorted(self._adapters)
            if (projected := self.policy.project(self._descriptors[operation_id]))
            is not None
        )
        return OperationCatalogSnapshot(
            policy_profile=self.policy.profile,
            policy_digest=self.policy.digest,
            operations=projected,
        )

    def inspect(self, operation_id: str) -> OperationDescriptor | None:
        descriptor = self._descriptors.get(operation_id)
        return None if descriptor is None else self.policy.project(descriptor)

    def search(self, request: OperationDiscoveryRequest) -> OperationDiscoveryResult:
        return discover_operations(self.snapshot(), request)

    def invoke(self, request: OperationRequest) -> OperationResult:
        return dispatch_operation(self, request)


__all__ = ["CatalogOperationCollector"]
