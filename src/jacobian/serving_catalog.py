"""Serving catalog of built-in declarations plus an optional SQLite overlay."""

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
    OperationCatalogError,
    OperationCheckerBinding,
    OperationDeclarationRecord,
    OperationSearchCard,
    VisibilityPolicy,
    omitted_packaged_operations,
    public_operation_descriptor,
)
from jacobian.operation_discovery import discover_operations
from jacobian.operation_locators import FamilyLocator, encode_locator
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.package_index import PackageIndex, PackageIndexEntry, load_package_index

_OPTIONAL_INDEX_FAMILIES = frozenset({"lean", "sat-smt"})


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
        self._omitted_packaged = _omitted_packaged_ids(index, overlay)

    @classmethod
    def open(
        cls,
        database_path: Path | None,
        policy: OperationVisibilityPolicy | None = None,
        *,
        expected_package_version: str | None = None,
    ) -> ServingCatalog:
        """Open built-in declarations, plus SQLite overlay when state exists."""

        visibility = policy or OperationVisibilityPolicy()
        overlay = None
        if database_path is not None:
            if database_path.exists() and not database_path.is_file():
                raise OperationCatalogError(
                    "STATE_UPDATE_REQUIRED: catalog state is unreadable; "
                    "run `jacobian update`"
                )
            if database_path.is_file():
                overlay = OperationCatalog(
                    database_path,
                    visibility,
                    expected_package_version=expected_package_version or __version__,
                )
        catalog = cls(load_package_index(), overlay, visibility)
        _reject_packaged_descriptor_mirrors(catalog.index, overlay)
        return catalog

    def inspect(self, operation_id: str) -> OperationDescriptor | None:
        entry = self._visible_index_entry(operation_id)
        if entry is not None:
            return self.policy.project(public_operation_descriptor(entry.descriptor()))
        if self.overlay is None:
            return None
        return self.overlay.inspect(operation_id)

    def declaration_record(
        self, operation_id: str
    ) -> OperationDeclarationRecord | None:
        entry = self._visible_index_entry(operation_id)
        if entry is not None:
            module = entry.module
            if entry.family is not None:
                module = encode_locator(FamilyLocator(family=entry.family))
            return OperationDeclarationRecord(
                operation_id=entry.operation_id,
                module=module,
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
        for entry in self.index.entries.values():
            if self._visible_index_entry(entry.operation_id) is None:
                continue
            projected = self.policy.project(
                public_operation_descriptor(entry.descriptor())
            )
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

    def _visible_index_entry(self, operation_id: str) -> PackageIndexEntry | None:
        if operation_id in self._omitted_packaged:
            return None
        return self.index.get(operation_id)

    def cards(self) -> tuple[OperationSearchCard, ...]:
        return tuple(
            OperationSearchCard.from_descriptor(descriptor)
            for descriptor in self.snapshot().operations
        )


def _omitted_packaged_ids(
    index: PackageIndex, overlay: OperationCatalog | None
) -> frozenset[str]:
    """Hide unavailable optional families and compile-omitted packaged IDs."""

    if overlay is None:
        return frozenset(
            entry.operation_id
            for entry in index.entries.values()
            if entry.family in _OPTIONAL_INDEX_FAMILIES
        )
    return omitted_packaged_operations(overlay.header.diagnostics)


def _reject_packaged_descriptor_mirrors(
    index: PackageIndex,
    overlay: OperationCatalog | None,
) -> None:
    """Refuse SQLite copies of packaged built-in descriptors."""

    if overlay is None:
        return
    for operation_id in index.entries:
        if overlay.declaration_record(operation_id) is not None:
            raise OperationCatalogError(
                "operation catalog overlay is stale; run `jacobian update`"
            )


__all__ = ["ServingCatalog"]
