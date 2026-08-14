"""Composition tests for explicit mathematical operation groups."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from jacobian.artifacts import ArtifactService
from jacobian.catalog_operation_collector import CatalogOperationCollector
from jacobian.contracts.operations import (
    OperationDescriptor,
)
from jacobian.domains.arithmetic import arithmetic_operations
from jacobian.domains.combinatorics import combinatorics_operations
from jacobian.domains.finite_sets import finite_set_operations
from jacobian.domains.number_theory import number_theory_operations
from jacobian.domains.sequences import sequence_operations
from jacobian.operation_binding import OperationBinder
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository

ALL_OPERATION_GROUPS = (
    arithmetic_operations(),
    combinatorics_operations(),
    finite_set_operations(),
    number_theory_operations(),
    sequence_operations(),
)


def _all_operation_ids() -> set[str]:
    ids: set[str] = set()
    for operations in ALL_OPERATION_GROUPS:
        for operation in operations:
            ids.add(operation.operation_id)
    return ids


def test_declared_groups_expose_operations() -> None:
    actual = _all_operation_ids()
    assert actual, "expected at least one declared operation"
    assert len(actual) == sum(len(operations) for operations in ALL_OPERATION_GROUPS)


def test_unique_ids_within_each_declaration_group() -> None:
    for operations in ALL_OPERATION_GROUPS:
        ids = [op.operation_id for op in operations]
        assert len(ids) == len(set(ids)), f"duplicates: {ids}"


def test_no_id_in_two_declaration_groups() -> None:
    seen: dict[str, str] = {}
    for group_index, operations in enumerate(ALL_OPERATION_GROUPS):
        for operation in operations:
            cap_id = operation.operation_id
            assert cap_id not in seen, (
                f"{cap_id!r} in both {seen[cap_id]!r} and group {group_index!r}"
            )
            seen[cap_id] = str(group_index)


@pytest.fixture
def service(tmp_path: Path) -> Iterator[CatalogOperationCollector]:
    store = ArtifactRepository(tmp_path / "state")
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    service = CatalogOperationCollector(store)
    installer = OperationBinder(store, schemas, artifacts)
    for operations in ALL_OPERATION_GROUPS:
        for adapter in installer.bind(operations).adapters:
            service.register(adapter)
    try:
        yield service
    finally:
        store.close()


def test_catalog_matches_installed_operations(
    service: CatalogOperationCollector,
) -> None:
    catalog_ids = {d.operation_id for d in service.snapshot().operations}
    expected = _all_operation_ids()
    assert catalog_ids == expected, (
        f"missing from catalog: {sorted(expected - catalog_ids)}\n"
        f"extra in catalog: {sorted(catalog_ids - expected)}"
    )
    by_id: dict[str, OperationDescriptor] = {
        d.operation_id: d for d in service.snapshot().operations
    }
    for operations in ALL_OPERATION_GROUPS:
        for operation in operations:
            desc = by_id[operation.operation_id]
            assert desc.version == operation.version
            assert desc.title == operation.title
            assert desc.description == operation.description
            assert desc.provider == "built-in"
            assert desc.tags == operation.tags
