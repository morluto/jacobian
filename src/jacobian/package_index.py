"""In-process inventory of built-in mathematical operations.

``math.find`` and ``math.run`` load ``InlineOperation`` declarations and family
discovery cards from source. There is no generated JSON schema dump.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

from jacobian.builtin_operation_modules import BUILTIN_OPERATION_MODULES
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiagnostic,
    OperationExample,
    OperationInputKind,
    OperationRequest,
)
from jacobian.family_catalog import family_index_payloads
from jacobian.inline_execution import (
    InlineOperationAdapter,
    inline_operation_descriptor,
)
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_catalog import OperationCatalogError, OperationSearchCard
from jacobian.operation_declarations import InlineOperation
from jacobian.operation_errors import OperationInvocationError
from jacobian.operation_projection import OperationProjection


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
    family: str | None = None
    read_only: bool = True
    accepted_input_kinds: tuple[OperationInputKind, ...] = (
        OperationInputKind.STRUCTURED_REQUEST,
    )
    accepted_artifact_types: tuple[str, ...] = ()
    produced_artifact_types: tuple[str, ...] = ()

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
            accepted_input_kinds=self.accepted_input_kinds,
            accepted_artifact_types=self.accepted_artifact_types,
            produced_artifact_types=self.produced_artifact_types,
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

    def cards(self) -> tuple[OperationSearchCard, ...]:
        return tuple(
            OperationSearchCard.from_descriptor(entry.descriptor())
            for entry in sorted(
                self.entries.values(), key=lambda item: item.operation_id
            )
        )

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
        if entry.family is not None:
            raise OperationCatalogError(
                f"family operation requires overlay catalog state: {operation_id}"
            )
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


def collect_family_index_entries() -> tuple[PackageIndexEntry, ...]:
    """Compile family discovery cards without constructing adapters."""

    entries: list[PackageIndexEntry] = []
    for payload in family_index_payloads():
        examples = tuple(
            OperationExample(
                name=str(example["name"]),
                description=str(example["description"]),
                input=dict(example["input"]),
            )
            for example in cast(list[dict[str, Any]], payload.get("examples") or ())
        )
        entries.append(
            PackageIndexEntry(
                operation_id=str(payload["operation_id"]),
                version=str(payload["version"]),
                title=str(payload["title"]),
                description=str(payload["description"]),
                tags=tuple(str(tag) for tag in payload["tags"]),
                examples=examples,
                input_schema=dict(payload["input_schema"]),
                output_schema=dict(payload["output_schema"]),
                module="",
                symbol="",
                family=str(payload["family"]),
                read_only=bool(payload["read_only"]),
                accepted_input_kinds=tuple(
                    OperationInputKind(kind)
                    for kind in cast(list[str], payload["accepted_input_kinds"])
                ),
                accepted_artifact_types=tuple(
                    str(uri)
                    for uri in cast(list[str], payload["accepted_artifact_types"])
                ),
                produced_artifact_types=tuple(
                    str(uri)
                    for uri in cast(list[str], payload["produced_artifact_types"])
                ),
            )
        )
    return tuple(sorted(entries, key=lambda item: item.operation_id))


def collect_package_index_entries() -> tuple[PackageIndexEntry, ...]:
    """Return built-in inline and family discovery cards."""

    combined = {
        entry.operation_id: entry
        for entry in (*collect_inline_index_entries(), *collect_family_index_entries())
    }
    return tuple(sorted(combined.values(), key=lambda item: item.operation_id))


def generate_package_index() -> PackageIndex:
    """Build the in-memory inventory from live declarations and family cards."""

    return PackageIndex(
        {entry.operation_id: entry for entry in collect_package_index_entries()}
    )


def load_package_index() -> PackageIndex:
    """Load built-in declarations so ``math.find`` / ``math.run`` can resolve IDs."""

    return generate_package_index()


class FamilyStateRequiredAdapter:
    """Fail closed when a family ID is run without overlay catalog state."""

    def __init__(self, descriptor: OperationDescriptor) -> None:
        self._descriptor = descriptor

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> None:
        raise OperationInvocationError(_family_state_required_diagnostic())

    def invoke(self, prepared: None) -> OperationProjection:
        raise OperationInvocationError(_family_state_required_diagnostic())


def _family_state_required_diagnostic() -> OperationDiagnostic:
    return OperationDiagnostic(
        code="STATE_INITIALIZATION_REQUIRED",
        stage="operation_resolution",
        message="Family operations require initialized catalog overlay state.",
        hint="Run `jacobian init` to create overlay catalog state, then retry.",
    )


class PackageIndexRegistry:
    """Resolve built-in inline operations without a binder or SQLite digest."""

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
            raise OperationCatalogError(f"unknown or hidden operation: {operation_id}")
        if entry.family is not None:
            adapter: OperationAdapter[Any] = FamilyStateRequiredAdapter(
                entry.descriptor()
            )
        else:
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
    "collect_family_index_entries",
    "collect_inline_index_entries",
    "collect_package_index_entries",
    "generate_package_index",
    "load_inline_operation",
    "load_package_index",
]
