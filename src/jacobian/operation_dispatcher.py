"""Catalog-backed lazy dispatch for selected built-in operations."""

from __future__ import annotations

from jacobian.contracts.operations import (
    OperationCatalogSnapshot,
    OperationDescriptor,
    OperationDiscoveryRequest,
    OperationDiscoveryResult,
    OperationRequest,
    OperationResult,
)
from jacobian.operation_catalog import OperationCatalog
from jacobian.operation_registry import OperationRegistry
from jacobian.operation_service import OperationPolicy, OperationService


class OperationDispatcher(OperationService):
    """Resolve a visible operation only when its first request arrives."""

    def __init__(self, catalog: OperationCatalog, registry: OperationRegistry) -> None:
        if not isinstance(catalog.policy, OperationPolicy):
            raise TypeError("operation dispatcher requires an OperationPolicy")
        super().__init__(registry.binder.store, policy=catalog.policy)
        self._catalog = catalog
        self._registry = registry

    def invoke(self, request: OperationRequest) -> OperationResult:
        if (
            request.operation_id not in self._adapters
            and self._catalog.inspect(request.operation_id) is not None
        ):
            self.register(self._registry.resolve(request.operation_id))
        return super().invoke(request)

    def discover(self, request: OperationDiscoveryRequest) -> OperationDiscoveryResult:
        return self._catalog.search(request)

    def inspect(self, operation_id: str) -> OperationDescriptor | None:
        return self._catalog.inspect(operation_id)

    def catalog(self) -> OperationCatalogSnapshot:
        return self._catalog.snapshot()

    def prepare_complete_binding(self) -> None:
        """Drop lazy adapters before the exceptional checker/resource cutover."""

        self._adapters.clear()
        self._descriptors.clear()


__all__ = ["OperationDispatcher"]
