"""Lazy resolution of one selected built-in mathematical operation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jacobian.builtin_operation_modules import (
    BUILTIN_OPERATION_MODULES,
    load_builtin_operation_module,
)
from jacobian.checker_operations import AuthorizedChecker
from jacobian.contracts.operations import OperationDescriptor
from jacobian.inline_execution import InlineOperationAdapter
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_binding import OperationBinder
from jacobian.operation_catalog import (
    OperationCatalog,
    OperationCatalogError,
    OperationCatalogView,
    OperationDeclarationRecord,
    exact_checker_declaration_digest,
    operation_declaration_digest,
    operation_declaration_digest_from_descriptor,
    public_operation_descriptor,
)
from jacobian.operation_declarations import InlineOperation, OperationDeclarations
from jacobian.operation_locators import FamilyLocator, ModuleLocator, decode_locator
from jacobian.package_index import PackageIndex, load_package_index
from jacobian.registry import CheckerRegistry
from jacobian.selected_operation_bindings import (
    ResourceOwner,
    SelectedOperationBinding,
)

if TYPE_CHECKING:
    from jacobian.runtime.execution import LazyControlPlane


class OperationRegistry:
    """Resolve persisted declarations through one runtime-local family table."""

    def __init__(
        self,
        catalog: OperationCatalogView,
        binder: OperationBinder,
        checkers: CheckerRegistry,
        resource_owner: ResourceOwner,
        *,
        control_plane: LazyControlPlane,
        package_index: PackageIndex | None = None,
    ) -> None:
        self.catalog = catalog
        self.binder = binder
        self.checkers = checkers
        self._control_plane = control_plane
        self._resource_owner = resource_owner
        self._package_index = (
            package_index if package_index is not None else load_package_index()
        )
        self._adapters: dict[str, OperationAdapter[Any]] = {}

    def close(self) -> None:
        """Discard cached adapters before the runtime closes their resources."""

        self._adapters.clear()

    def _resolve_package_index(self, operation_id: str) -> OperationAdapter[Any]:
        adapter: OperationAdapter[Any] = InlineOperationAdapter(
            self._package_index.load(operation_id)
        )
        self._adapters[operation_id] = adapter
        return adapter

    def resolve(self, operation_id: str) -> OperationAdapter[Any]:
        cached = self._adapters.get(operation_id)
        if cached is not None:
            return cached
        if self._package_index.contains(operation_id):
            entry = self._package_index.get(operation_id)
            if entry is not None and entry.family is not None:
                descriptor = self.catalog.inspect(operation_id)
                record = self.catalog.declaration_record(operation_id)
                if descriptor is None or record is None:
                    raise OperationCatalogError(
                        f"unknown or hidden operation: {operation_id}"
                    )
                return self._resolve_selected_family(
                    operation_id,
                    descriptor,
                    record,
                    FamilyLocator(family=entry.family),
                )
            return self._resolve_package_index(operation_id)
        descriptor = self.catalog.inspect(operation_id)
        record = self.catalog.declaration_record(operation_id)
        if descriptor is None or record is None:
            raise OperationCatalogError(f"unknown or hidden operation: {operation_id}")

        locator = decode_locator(record.module)
        if isinstance(locator, FamilyLocator):
            return self._resolve_selected_family(
                operation_id, descriptor, record, locator
            )
        return self._resolve_declaration_module(
            operation_id, descriptor, record, locator
        )

    def _resolve_declaration_module(
        self,
        operation_id: str,
        descriptor: OperationDescriptor,
        record: OperationDeclarationRecord,
        locator: ModuleLocator,
    ) -> OperationAdapter[Any]:
        if locator.module not in {
            module_name for module_name, _factory_name in BUILTIN_OPERATION_MODULES
        }:
            raise OperationCatalogError(
                "operation catalog locator is stale; run `jacobian update`: "
                f"{operation_id}"
            )
        _module_name, operations, checker_declarations = load_builtin_operation_module(
            locator.module
        )
        matches = tuple(
            operation
            for operation in operations
            if operation.operation_id == operation_id
        )
        if not matches and any(
            declaration.verification_operation_id == operation_id
            for declaration in checker_declarations
        ):
            return self._resolve_exact_verifier(
                operation_id,
                descriptor,
                record,
                operations,
                checker_declarations,
            )
        if len(matches) != 1:
            raise OperationCatalogError(
                f"operation locator did not resolve exactly once: {operation_id}"
            )
        declaration = matches[0]
        if declaration.version != descriptor.version:
            raise OperationCatalogError(
                "operation declaration version changed; run `jacobian update`: "
                f"{operation_id}"
            )
        if (
            record.declaration_digest != "package-index"
            and operation_declaration_digest(declaration) != record.declaration_digest
        ):
            raise OperationCatalogError(
                f"operation declaration changed; run `jacobian update`: {operation_id}"
            )
        if isinstance(declaration, InlineOperation):
            adapter: OperationAdapter[Any] = InlineOperationAdapter(declaration)
        else:
            bound = self.binder.bind(operations)
            adapter = next(
                candidate
                for candidate in bound.adapters
                if candidate.descriptor.operation_id == operation_id
            )
        if public_operation_descriptor(adapter.descriptor) != descriptor:
            raise OperationCatalogError(
                f"operation schema changed; run `jacobian update`: {operation_id}"
            )
        self._adapters[operation_id] = adapter
        return adapter

    def _resolve_selected_family(
        self,
        operation_id: str,
        descriptor: OperationDescriptor,
        record: OperationDeclarationRecord,
        locator: FamilyLocator,
    ) -> OperationAdapter[Any]:
        family = next(
            (
                family
                for family in self._control_plane.families
                if family.spec.origin == locator.family
            ),
            None,
        )
        if family is None:
            raise OperationCatalogError(
                f"selected operation family is unavailable: {operation_id}"
            )
        binding = family.bind(operation_id, descriptor)
        if binding is None:
            raise OperationCatalogError(
                f"selected operation binder is missing: {operation_id}"
            )
        if record.declaration_digest != "package-index":
            self._validate_selected_binding(operation_id, descriptor, record, binding)
        for resource in binding.resources:
            self._resource_owner.own(resource)
        adapter = binding.adapter
        self._adapters[operation_id] = adapter
        return adapter

    @staticmethod
    def _validate_selected_binding(
        operation_id: str,
        descriptor: OperationDescriptor,
        record: OperationDeclarationRecord,
        binding: SelectedOperationBinding,
    ) -> None:
        if (
            operation_declaration_digest_from_descriptor(descriptor)
            != record.declaration_digest
        ):
            raise OperationCatalogError(
                f"operation declaration changed; run `jacobian update`: {operation_id}"
            )
        if public_operation_descriptor(binding.adapter.descriptor) != descriptor:
            raise OperationCatalogError(
                f"operation schema changed; run `jacobian update`: {operation_id}"
            )

    def _resolve_exact_verifier(
        self,
        operation_id: str,
        descriptor: OperationDescriptor,
        record: OperationDeclarationRecord,
        operations: OperationDeclarations,
        checker_declarations: tuple[AuthorizedChecker, ...],
    ) -> OperationAdapter[Any]:
        from jacobian.exact_domain_verification import bind_selected_exact_verification

        checker_declaration = next(
            declaration
            for declaration in checker_declarations
            if declaration.verification_operation_id == operation_id
        )
        if (
            exact_checker_declaration_digest(checker_declaration, descriptor)
            != record.declaration_digest
        ):
            raise OperationCatalogError(
                f"operation declaration changed; run `jacobian update`: {operation_id}"
            )
        adapter = bind_selected_exact_verification(
            catalog=_sqlite_catalog(self.catalog, operation_id),
            operation_id=operation_id,
            operations=operations,
            declarations=checker_declarations,
            binder=self.binder,
            verification=self._control_plane.verification,
            checkers=self.checkers,
        )
        if public_operation_descriptor(adapter.descriptor) != descriptor:
            raise OperationCatalogError(
                f"operation schema changed; run `jacobian update`: {operation_id}"
            )
        self._adapters[operation_id] = adapter
        return adapter


def _sqlite_catalog(
    catalog: OperationCatalogView, operation_id: str
) -> OperationCatalog:
    if isinstance(catalog, OperationCatalog):
        return catalog
    overlay = getattr(catalog, "overlay", None)
    if isinstance(overlay, OperationCatalog):
        return overlay
    raise OperationCatalogError(
        f"exact verifier requires overlay catalog state: {operation_id}"
    )


__all__ = ["OperationRegistry"]
