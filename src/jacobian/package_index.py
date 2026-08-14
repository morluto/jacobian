"""Read-only package index for built-in mathematical operations.

The index is generated from ``InlineOperation`` declarations and family cards
at wheel build. Editable checkouts reconstruct it in memory when the packaged
JSON is absent.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from importlib import import_module, resources
from typing import Any, cast

from jacobian.builtin_operation_modules import BUILTIN_OPERATION_MODULES
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationExample,
)
from jacobian.inline_execution import (
    InlineOperationAdapter,
    inline_operation_descriptor,
)
from jacobian.operation_adapters import OperationAdapter
from jacobian.operation_catalog import OperationCatalogError, OperationSearchCard
from jacobian.operation_declarations import InlineOperation

_PACKAGE_INDEX_RESOURCE = "inline_index.json"


@dataclass(frozen=True, slots=True)
class PackageIndexEntry:
    """One packaged inline operation and the symbol that reconstructs it."""

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

    def descriptor(self) -> OperationDescriptor:
        return OperationDescriptor(
            operation_id=self.operation_id,
            version=self.version,
            title=self.title,
            description=self.description,
            provider="built-in",
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            read_only=self.family is None,
            tags=self.tags,
            produced_artifact_types=(),
            examples=self.examples,
        )

    def as_json(self) -> dict[str, Any]:
        payload = {
            "operation_id": self.operation_id,
            "version": self.version,
            "title": self.title,
            "description": self.description,
            "tags": list(self.tags),
            "examples": [
                {
                    "name": example.name,
                    "description": example.description,
                    "input": dict(example.input),
                }
                for example in self.examples
            ],
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "module": self.module,
            "symbol": self.symbol,
        }
        if self.family is not None:
            payload["family"] = self.family
        return payload


@dataclass(frozen=True, slots=True)
class PackageIndex:
    """Immutable lookup of packaged inline operations."""

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

    # Generate-only import: serving loads JSON and must not import family contracts.
    from jacobian.family_catalog import family_index_payloads

    entries: list[PackageIndexEntry] = []
    for payload in family_index_payloads():
        entries.append(
            PackageIndexEntry(
                operation_id=str(payload["operation_id"]),
                version=str(payload["version"]),
                title=str(payload["title"]),
                description=str(payload["description"]),
                tags=tuple(str(tag) for tag in payload["tags"]),
                examples=(),
                input_schema=dict(payload["input_schema"]),
                output_schema=dict(payload["output_schema"]),
                module="",
                symbol="",
                family=str(payload["family"]),
            )
        )
    return tuple(sorted(entries, key=lambda item: item.operation_id))


def collect_package_index_entries() -> tuple[PackageIndexEntry, ...]:
    """Return packaged inline and family discovery cards."""

    combined = {
        entry.operation_id: entry
        for entry in (*collect_inline_index_entries(), *collect_family_index_entries())
    }
    return tuple(sorted(combined.values(), key=lambda item: item.operation_id))


def generate_package_index() -> PackageIndex:
    """Build the in-memory index from live inline declarations."""

    return PackageIndex(
        {entry.operation_id: entry for entry in collect_package_index_entries()}
    )


def load_package_index() -> PackageIndex:
    """Load the packaged index, reconstructing it when the JSON is absent."""

    packaged = _read_packaged_index()
    if packaged is not None:
        return packaged
    return generate_package_index()


def write_package_index(path: Any) -> None:
    """Write the generated index JSON for a wheel or local package data file."""

    payload = [entry.as_json() for entry in collect_package_index_entries()]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class PackageIndexRegistry:
    """Resolve packaged inline operations without a binder or SQLite digest."""

    def __init__(self, index: PackageIndex) -> None:
        self.index = index
        self.binder = None
        self._adapters: dict[str, OperationAdapter[Any]] = {}

    def resolve(self, operation_id: str) -> OperationAdapter[Any]:
        cached = self._adapters.get(operation_id)
        if cached is not None:
            return cached
        try:
            operation = self.index.load(operation_id)
        except KeyError as exc:
            raise OperationCatalogError(
                f"unknown or hidden operation: {operation_id}"
            ) from exc
        adapter: OperationAdapter[Any] = InlineOperationAdapter(operation)
        self._adapters[operation_id] = adapter
        return adapter

    def close(self) -> None:
        self._adapters.clear()


def load_inline_operation(entry: PackageIndexEntry) -> InlineOperation[Any, Any]:
    """Load one inline operation from its packaged module and symbol."""

    module = import_module(entry.module)
    target = getattr(module, entry.symbol)
    if isinstance(target, InlineOperation):
        if target.operation_id != entry.operation_id:
            raise RuntimeError(
                "packaged inline symbol does not match operation_id: "
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
            "packaged inline locator did not resolve exactly once: "
            f"{entry.operation_id}"
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


def _read_packaged_index() -> PackageIndex | None:
    try:
        payload = resources.files("jacobian.data").joinpath(_PACKAGE_INDEX_RESOURCE)
        if not payload.is_file():
            return None
        raw = json.loads(payload.read_text(encoding="utf-8"))
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return None
    if not isinstance(raw, list):
        raise RuntimeError("packaged inline index must be a JSON array")
    entries = tuple(_entry_from_json(item) for item in raw)
    return PackageIndex({entry.operation_id: entry for entry in entries})


def _entry_from_json(item: object) -> PackageIndexEntry:
    if not isinstance(item, dict):
        raise RuntimeError("packaged inline index entries must be objects")
    payload = cast(dict[str, Any], item)
    examples = tuple(
        OperationExample(
            name=str(example["name"]),
            description=str(example["description"]),
            input=dict(example["input"]),
        )
        for example in cast(list[dict[str, Any]], payload["examples"])
    )
    return PackageIndexEntry(
        operation_id=str(payload["operation_id"]),
        version=str(payload["version"]),
        title=str(payload["title"]),
        description=str(payload["description"]),
        tags=tuple(str(tag) for tag in cast(list[Any], payload["tags"])),
        examples=examples,
        input_schema=dict(payload["input_schema"]),
        output_schema=dict(payload["output_schema"]),
        module=str(payload.get("module") or ""),
        symbol=str(payload.get("symbol") or ""),
        family=(str(payload["family"]) if payload.get("family") is not None else None),
    )


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
    "write_package_index",
]
