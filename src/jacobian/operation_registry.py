"""Lazy resolution of one selected built-in mathematical operation."""

from __future__ import annotations

from typing import Any

from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_binding import OperationBinder
from jacobian.operation_catalog import (
    OperationCatalog,
    OperationCatalogError,
    operation_declaration_digest,
)
from jacobian.portfolio.builtin import load_builtin_operation_module


class OperationRegistry:
    """Import, verify, and cache only selected built-in declarations."""

    def __init__(self, catalog: OperationCatalog, binder: OperationBinder) -> None:
        self.catalog = catalog
        self.binder = binder
        self._adapters: dict[str, OperationAdapter[Any]] = {}

    def resolve(self, operation_id: str) -> OperationAdapter[Any]:
        cached = self._adapters.get(operation_id)
        if cached is not None:
            return cached
        descriptor = self.catalog.inspect(operation_id)
        record = self.catalog.declaration_record(operation_id)
        if descriptor is None or record is None:
            raise OperationCatalogError(f"unknown or hidden operation: {operation_id}")
        try:
            _module_name, operations, _checkers = load_builtin_operation_module(
                record.module
            )
        except ValueError as exc:
            raise OperationCatalogError(
                f"operation {operation_id} is not a declared built-in operation"
            ) from exc
        matches = tuple(
            operation
            for operation in operations
            if operation.operation_id == operation_id
        )
        if len(matches) != 1:
            raise OperationCatalogError(
                f"operation locator did not resolve exactly once: {operation_id}"
            )
        declaration = matches[0]
        if declaration.version != descriptor.version:
            raise OperationCatalogError(
                f"operation declaration version changed; run `jacobian update`: {operation_id}"
            )
        if operation_declaration_digest(declaration) != record.declaration_digest:
            raise OperationCatalogError(
                f"operation declaration changed; run `jacobian update`: {operation_id}"
            )
        adapter = self.binder.bind((declaration,)).adapters[0]
        if adapter.descriptor.model_dump(mode="json") != descriptor.model_dump(
            mode="json"
        ):
            raise OperationCatalogError(
                f"operation schema changed; run `jacobian update`: {operation_id}"
            )
        self._adapters[operation_id] = adapter
        return adapter


__all__ = ["OperationRegistry"]
