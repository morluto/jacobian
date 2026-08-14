"""Immutable catalog of packaged inline operations."""

from __future__ import annotations

from jacobian.contracts.operations import (
    OperationCatalogSnapshot,
    OperationDescriptor,
    OperationDiscoveryRequest,
    OperationDiscoveryResult,
)
from jacobian.operation_discovery import discover_operations
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.package_index import PackageIndex, load_package_index


class ServingCatalog:
    """Direct package-index view with no overlay or state directory."""

    def __init__(self, index: PackageIndex, policy: OperationVisibilityPolicy) -> None:
        self.index = index
        self.policy = policy

    @classmethod
    def open(cls, *, policy: OperationVisibilityPolicy | None = None) -> ServingCatalog:
        return cls(load_package_index(), policy or OperationVisibilityPolicy())

    def inspect(self, operation_id: str) -> OperationDescriptor | None:
        entry = self.index.get(operation_id)
        if entry is None:
            return None
        return self.policy.project(entry.descriptor())

    def search(self, request: OperationDiscoveryRequest) -> OperationDiscoveryResult:
        return discover_operations(self.snapshot(), request)

    def snapshot(self) -> OperationCatalogSnapshot:
        operations = tuple(
            descriptor
            for entry in self.index.entries.values()
            if (descriptor := self.policy.project(entry.descriptor())) is not None
        )
        return OperationCatalogSnapshot(
            policy_profile=self.policy.profile,
            policy_digest=self.policy.digest,
            operations=tuple(sorted(operations, key=lambda item: item.operation_id)),
        )


__all__ = ["ServingCatalog"]
