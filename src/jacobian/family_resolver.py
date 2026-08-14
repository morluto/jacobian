"""Runtime-local family resolvers: shared resources once, one adapter cache."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from jacobian.contracts.operations import OperationDescriptor
from jacobian.operation_adapters import OperationAdapter
from jacobian.selected_operation_bindings import SelectedOperationBinding


class FamilyResolver:
    """Bind one family ID through cached adapters and a close boundary."""

    def __init__(
        self,
        family: str,
        bind: Callable[[str, OperationDescriptor], SelectedOperationBinding | None],
        *,
        close_resources: Callable[[], None] | None = None,
    ) -> None:
        self.family = family
        self._bind = bind
        self._close_resources = close_resources
        self._adapters: dict[str, OperationAdapter[Any]] = {}
        self._lock = Lock()

    def resolve(
        self, operation_id: str, descriptor: OperationDescriptor
    ) -> SelectedOperationBinding | None:
        cached = self._adapters.get(operation_id)
        if cached is not None:
            return SelectedOperationBinding(cached)
        with self._lock:
            cached = self._adapters.get(operation_id)
            if cached is not None:
                return SelectedOperationBinding(cached)
            binding = self._bind(operation_id, descriptor)
            if binding is not None:
                self._adapters[operation_id] = binding.adapter
            return binding

    def close(self) -> None:
        self._adapters.clear()
        if self._close_resources is not None:
            self._close_resources()


@dataclass(slots=True)
class FamilyResolverTable:
    """One resolver per family, constructed with the runtime's shared resources."""

    resolvers: dict[str, FamilyResolver] = field(default_factory=dict)

    def get(self, family: str) -> FamilyResolver | None:
        return self.resolvers.get(family)

    def close(self) -> None:
        for resolver in self.resolvers.values():
            resolver.close()
        self.resolvers.clear()


__all__ = ["FamilyResolver", "FamilyResolverTable"]
