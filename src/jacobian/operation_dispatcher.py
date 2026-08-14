"""Catalog-backed lazy dispatch for selected built-in operations."""

from __future__ import annotations

from threading import Lock
from typing import Any, Protocol

from jacobian.contracts.operations import (
    OperationCatalogSnapshot,
    OperationDescriptor,
    OperationDiscoveryRequest,
    OperationDiscoveryResult,
    OperationRequest,
    OperationResult,
)
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_dispatch import dispatch_operation, register_operation
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.serving_catalog import ServingCatalog


class OperationResolver(Protocol):
    """Resolve one visible operation adapter and participate in shutdown."""

    binder: Any

    def resolve(self, operation_id: str) -> OperationAdapter[Any]: ...

    def close(self) -> None: ...


class OperationDispatcher:
    """Resolve a visible operation only when its first request arrives."""

    def __init__(self, catalog: ServingCatalog, registry: OperationResolver) -> None:
        if not isinstance(catalog.policy, OperationVisibilityPolicy):
            raise TypeError(
                "operation dispatcher requires an OperationVisibilityPolicy"
            )
        binder = registry.binder
        self.store = None if binder is None else binder.store
        self.policy = catalog.policy
        self._adapters: dict[str, OperationAdapter[Any]] = {}
        self._descriptors: dict[str, OperationDescriptor] = {}
        self._catalog = catalog
        self._registry = registry
        self._registration_lock = Lock()

    def register(self, adapter: OperationAdapter[Any]) -> None:
        register_operation(adapter, self._adapters, self._descriptors)

    def invoke(self, request: OperationRequest) -> OperationResult:
        if request.operation_id not in self._adapters:
            with self._registration_lock:
                if (
                    request.operation_id not in self._adapters
                    and self._catalog.inspect(request.operation_id) is not None
                ):
                    self.register(self._registry.resolve(request.operation_id))
        return dispatch_operation(self, request)

    def search(self, request: OperationDiscoveryRequest) -> OperationDiscoveryResult:
        return self._catalog.search(request)

    def inspect(self, operation_id: str) -> OperationDescriptor | None:
        return self._catalog.inspect(operation_id)

    def snapshot(self) -> OperationCatalogSnapshot:
        return self._catalog.snapshot()

    def close(self) -> None:
        self._registry.close()


__all__ = ["OperationDispatcher"]
