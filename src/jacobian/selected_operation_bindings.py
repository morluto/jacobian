"""Fixed selected-operation binding seams used by one execution runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from jacobian.contracts.operations import OperationDescriptor
from jacobian.operation_adapters import OperationAdapter


@dataclass(frozen=True, slots=True)
class SelectedOperationBinding:
    """One adapter and resources acquired while constructing that adapter."""

    adapter: OperationAdapter[Any]
    resources: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class SelectedFamilySpec:
    """Immutable family ownership metadata used at catalog compilation."""

    origin: str
    operation_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class RuntimeSelectedFamily:
    """One family binder captured by a single execution runtime."""

    spec: SelectedFamilySpec
    bind: Callable[[str, OperationDescriptor], SelectedOperationBinding | None]


class ResourceOwner(Protocol):
    """Runtime boundary that owns closeable selected-operation resources."""

    def own(self, resource: object) -> None: ...


__all__ = [
    "ResourceOwner",
    "RuntimeSelectedFamily",
    "SelectedFamilySpec",
    "SelectedOperationBinding",
]
