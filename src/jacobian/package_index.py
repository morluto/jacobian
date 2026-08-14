"""In-process inventory of built-in mathematical operations.

``math.find`` and ``math.run`` load ``InlineOperation`` declarations and family
discovery cards from source. There is no generated JSON schema dump.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from jacobian.builtin_operation_modules import BUILTIN_OPERATION_MODULES
from jacobian.contracts.operations import OperationDescriptor, OperationExample
from jacobian.inline_execution import (
    InlineOperationAdapter,
    inline_operation_descriptor,
)
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_declarations import InlineOperation


@dataclass(frozen=True, slots=True)
class PackageIndexEntry:
    """One built-in operation and the symbol that reconstructs it."""

    operation_id: str
    version: str
    title: str
    description: str
    tags: tuple[str, ...]
    examples: tuple[OperationExample, ...]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    module: str
    symbol: str
    read_only: bool = True

    def descriptor(self) -> OperationDescriptor:
        return OperationDescriptor(
            operation_id=self.operation_id,
            version=self.version,
            title=self.title,
            description=self.description,
            provider="built-in",
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            read_only=self.read_only,
            tags=self.tags,
            examples=self.examples,
        )


@dataclass(frozen=True, slots=True)
class PackageIndex:
    """Immutable lookup of built-in operations for search and run."""

    entries: dict[str, PackageIndexEntry]

    def get(self, operation_id: str) -> PackageIndexEntry | None:
        return self.entries.get(operation_id)

    def contains(self, operation_id: str) -> bool:
        return operation_id in self.entries

    def descriptors(self) -> tuple[OperationDescriptor, ...]:
        return tuple(
            entry.descriptor()
            for entry in sorted(
                self.entries.values(), key=lambda item: item.operation_id
            )
        )

    def load(self, operation_id: str) -> InlineOperation[Any, Any]:
        entry = self.entries.get(operation_id)
        if entry is None:
            raise KeyError(operation_id)
        return load_inline_operation(entry)


def collect_inline_index_entries() -> tuple[PackageIndexEntry, ...]:
    """Walk declaration modules without constructing operation adapters."""

    entries: list[PackageIndexEntry] = []
    for declared_module, factory_name in BUILTIN_OPERATION_MODULES:
        module = import_module(declared_module)
        factory = getattr(module, factory_name)
        operations = factory()
        for operation in operations:
            if not isinstance(operation, InlineOperation):
                continue
            descriptor = inline_operation_descriptor(operation)
            locator_module, symbol = _locator_for(module, factory_name, operation)
            entries.append(
                PackageIndexEntry(
                    operation_id=operation.operation_id,
                    version=operation.version,
                    title=operation.title,
                    description=operation.description,
                    tags=operation.tags,
                    examples=descriptor.examples,
                    input_schema=descriptor.input_schema,
                    output_schema=descriptor.output_schema,
                    module=locator_module,
                    symbol=symbol,
                )
            )
    return tuple(sorted(entries, key=lambda item: item.operation_id))


def collect_package_index_entries() -> tuple[PackageIndexEntry, ...]:
    """Return the immutable built-in inline declarations."""

    return collect_inline_index_entries()


def generate_package_index() -> PackageIndex:
    """Build the in-memory inventory from live inline declarations."""

    return PackageIndex(
        {entry.operation_id: entry for entry in collect_package_index_entries()}
    )


def load_package_index() -> PackageIndex:
    """Load built-in declarations so ``math.find`` / ``math.run`` can resolve IDs."""

    return generate_package_index()


class PackageIndexRegistry:
    """Resolve built-in inline operations without a binder or persistence digest."""

    def __init__(self, index: PackageIndex) -> None:
        self.index = index
        self.binder = None
        self._adapters: dict[str, OperationAdapter[Any]] = {}

    def resolve(self, operation_id: str) -> OperationAdapter[Any]:
        cached = self._adapters.get(operation_id)
        if cached is not None:
            return cached
        entry = self.index.get(operation_id)
        if entry is None:
            raise KeyError(operation_id)
        adapter = InlineOperationAdapter(load_inline_operation(entry))
        self._adapters[operation_id] = adapter
        return adapter

    def close(self) -> None:
        self._adapters.clear()


def load_inline_operation(entry: PackageIndexEntry) -> InlineOperation[Any, Any]:
    """Load one inline operation from its declaration module and symbol."""

    module = import_module(entry.module)
    target = getattr(module, entry.symbol)
    if isinstance(target, InlineOperation):
        if target.operation_id != entry.operation_id:
            raise RuntimeError(
                "inline declaration symbol does not match operation_id: "
                f"{entry.operation_id}"
            )
        return target
    operations = target() if callable(target) else target
    matches = tuple(
        operation
        for operation in operations
        if isinstance(operation, InlineOperation)
        and operation.operation_id == entry.operation_id
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"inline locator did not resolve exactly once: {entry.operation_id}"
        )
    return matches[0]


def _locator_for(
    module: Any, factory_name: str, operation: InlineOperation[Any, Any]
) -> tuple[str, str]:
    factory = getattr(module, factory_name, None)
    search = [module]
    defining = getattr(factory, "__module__", None)
    if defining and defining != module.__name__:
        search.append(import_module(defining))
    for candidate in search:
        for name, value in vars(candidate).items():
            if (
                isinstance(value, InlineOperation)
                and value.operation_id == operation.operation_id
            ):
                return candidate.__name__, name
    for loaded_name, loaded in list(sys.modules.items()):
        if loaded is None or not loaded_name.startswith("jacobian.domains"):
            continue
        for name, value in vars(loaded).items():
            if (
                isinstance(value, InlineOperation)
                and value.operation_id == operation.operation_id
            ):
                return loaded_name, name
    return module.__name__, factory_name


__all__ = [
    "PackageIndex",
    "PackageIndexEntry",
    "PackageIndexRegistry",
    "collect_inline_index_entries",
    "collect_package_index_entries",
    "generate_package_index",
    "load_inline_operation",
    "load_package_index",
]
