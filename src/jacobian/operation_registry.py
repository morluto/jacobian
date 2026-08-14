"""Lazy resolution of one selected built-in mathematical operation."""

from __future__ import annotations

from typing import Any

from jacobian.builtin_operation_modules import (
    BUILTIN_OPERATION_MODULES,
    load_builtin_operation_module,
)
from jacobian.checker_operations import AuthorizedChecker
from jacobian.contracts.operations import OperationDescriptor
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_binding import OperationBinder
from jacobian.operation_catalog import (
    OperationCatalog,
    OperationCatalogError,
    OperationDeclarationRecord,
    exact_checker_declaration_digest,
    operation_declaration_digest,
    operation_declaration_digest_from_descriptor,
    public_operation_descriptor,
)
from jacobian.operation_declarations import OperationDeclarations
from jacobian.registry import CheckerRegistry
from jacobian.selected_operation_bindings import (
    ResourceOwner,
    RuntimeSelectedFamily,
    SelectedOperationBinding,
)
from jacobian.verification.service import VerificationService


class OperationRegistry:
    """Resolve persisted declarations through one runtime-local family table."""

    def __init__(
        self,
        catalog: OperationCatalog,
        binder: OperationBinder,
        verification: VerificationService,
        checkers: CheckerRegistry,
        selected_families: tuple[RuntimeSelectedFamily, ...],
        resource_owner: ResourceOwner,
    ) -> None:
        self.catalog = catalog
        self.binder = binder
        self.verification = verification
        self.checkers = checkers
        self._selected_families = selected_families
        self._resource_owner = resource_owner
        self._adapters: dict[str, OperationAdapter[Any]] = {}

    def close(self) -> None:
        """Discard cached adapters before the runtime closes their resources."""

        self._adapters.clear()

    def resolve(self, operation_id: str) -> OperationAdapter[Any]:
        cached = self._adapters.get(operation_id)
        if cached is not None:
            return cached
        descriptor = self.catalog.inspect(operation_id)
        record = self.catalog.declaration_record(operation_id)
        if descriptor is None or record is None:
            raise OperationCatalogError(f"unknown or hidden operation: {operation_id}")

        if record.module.startswith("family:"):
            return self._resolve_selected_family(operation_id, descriptor, record)

        if record.module not in {
            module_name for module_name, _factory_name in BUILTIN_OPERATION_MODULES
        }:
            raise OperationCatalogError(
                "operation catalog locator is stale; run `jacobian update`: "
                f"{operation_id}"
            )
        _module_name, operations, checker_declarations = load_builtin_operation_module(
            record.module
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
        if operation_declaration_digest(declaration) != record.declaration_digest:
            raise OperationCatalogError(
                f"operation declaration changed; run `jacobian update`: {operation_id}"
            )
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
    ) -> OperationAdapter[Any]:
        family = next(
            (
                family
                for family in self._selected_families
                if family.spec.origin == record.module
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
            catalog=self.catalog,
            operation_id=operation_id,
            operations=operations,
            declarations=checker_declarations,
            binder=self.binder,
            verification=self.verification,
            checkers=self.checkers,
        )
        if public_operation_descriptor(adapter.descriptor) != descriptor:
            raise OperationCatalogError(
                f"operation schema changed; run `jacobian update`: {operation_id}"
            )
        self._adapters[operation_id] = adapter
        return adapter


__all__ = ["OperationRegistry"]
