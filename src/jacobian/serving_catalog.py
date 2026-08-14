"""Serving catalog that prefers the packaged inline index over SQLite copies."""

from __future__ import annotations

from pathlib import Path

from jacobian import __version__
from jacobian.contracts.operations import (
    OperationCatalogSnapshot,
    OperationDescriptor,
    OperationDiscoveryRequest,
    OperationDiscoveryResult,
)
from jacobian.operation_catalog import (
    OperationCatalog,
    OperationCheckerBinding,
    OperationDeclarationRecord,
    OperationSearchCard,
    VisibilityPolicy,
    public_operation_descriptor,
)
from jacobian.operation_discovery import discover_operations
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.package_index import PackageIndex, load_package_index


class ServingCatalog:
    """Merge packaged inline descriptors with an optional SQLite overlay."""

    def __init__(
        self,
        index: PackageIndex,
        overlay: OperationCatalog | None,
        policy: VisibilityPolicy,
    ) -> None:
        self.index = index
        self.overlay = overlay
        self.policy = policy

    @classmethod
    def open(
        cls,
        database_path: Path | None,
        policy: OperationVisibilityPolicy | None = None,
        *,
        expected_package_version: str | None = None,
    ) -> ServingCatalog:
        """Open the packaged index, plus SQLite overlay when state exists."""

        visibility = policy or OperationVisibilityPolicy()
        overlay = None
        if database_path is not None and database_path.is_file():
            overlay = OperationCatalog(
                database_path,
                visibility,
                expected_package_version=expected_package_version or __version__,
            )
        return cls(load_package_index(), overlay, visibility)

    def inspect(self, operation_id: str) -> OperationDescriptor | None:
        entry = self.index.get(operation_id)
        if entry is not None:
            return self.policy.project(public_operation_descriptor(entry.descriptor()))
        if self.overlay is None:
            return None
        return self.overlay.inspect(operation_id)

    def declaration_record(
        self, operation_id: str
    ) -> OperationDeclarationRecord | None:
        entry = self.index.get(operation_id)
        if entry is not None:
            return OperationDeclarationRecord(
                operation_id=entry.operation_id,
                module=entry.module,
                declaration_digest="package-index",
            )
        if self.overlay is None:
            return None
        return self.overlay.declaration_record(operation_id)

    def checker_binding(self, operation_id: str) -> OperationCheckerBinding | None:
        if self.overlay is None:
            return None
        return self.overlay.checker_binding(operation_id)

    def checker_bindings(
        self, operation_id: str
    ) -> tuple[OperationCheckerBinding, ...]:
        if self.overlay is None:
            return ()
        return self.overlay.checker_bindings(operation_id)

    def search(self, request: OperationDiscoveryRequest) -> OperationDiscoveryResult:
        return discover_operations(self.snapshot(), request)

    def snapshot(self) -> OperationCatalogSnapshot:
        descriptors: dict[str, OperationDescriptor] = {}
        for candidate in self.index.descriptors():
            projected = self.policy.project(public_operation_descriptor(candidate))
            if projected is not None:
                descriptors[projected.operation_id] = projected
        if self.overlay is not None:
            for descriptor in self.overlay.snapshot().operations:
                descriptors.setdefault(descriptor.operation_id, descriptor)
        operations = tuple(
            descriptors[operation_id] for operation_id in sorted(descriptors)
        )
        return OperationCatalogSnapshot(
            policy_profile=self.policy.profile,
            policy_digest=self.policy.digest,
            operations=operations,
        )

    def cards(self) -> tuple[OperationSearchCard, ...]:
        return tuple(
            OperationSearchCard.from_descriptor(descriptor)
            for descriptor in self.snapshot().operations
        )


__all__ = ["ServingCatalog"]
