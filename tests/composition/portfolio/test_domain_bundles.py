"""Composition tests for explicit mathematical operation groups."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from jacobian.artifacts import ArtifactService
from jacobian.contracts.operations import (
    OperationDescriptor,
)
from jacobian.domains.arithmetic import build_arithmetic_bundle
from jacobian.domains.combinatorics import build_combinatorics_bundle
from jacobian.domains.finite_sets import build_finite_set_bundle
from jacobian.domains.number_theory import build_number_theory_bundle
from jacobian.domains.sequences import build_sequence_bundle
from jacobian.operation_binding import OperationBinder
from jacobian.operation_service import OperationService
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository

ALL_OPERATION_GROUPS = (
    build_arithmetic_bundle(),
    build_combinatorics_bundle(),
    build_finite_set_bundle(),
    build_number_theory_bundle(),
    build_sequence_bundle(),
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


def test_unique_ids_within_each_bundle() -> None:
    for operations in ALL_OPERATION_GROUPS:
        ids = [op.operation_id for op in operations]
        assert len(ids) == len(set(ids)), f"duplicates: {ids}"


def test_no_id_in_two_bundles() -> None:
    seen: dict[str, str] = {}
    for group_index, operations in enumerate(ALL_OPERATION_GROUPS):
        for operation in operations:
            cap_id = operation.operation_id
            assert cap_id not in seen, (
                f"{cap_id!r} in both {seen[cap_id]!r} and group {group_index!r}"
            )
            seen[cap_id] = str(group_index)


@pytest.fixture
def service(tmp_path: Path) -> Iterator[OperationService]:
    store = ArtifactRepository(tmp_path / "state")
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    service = OperationService(store)
    installer = OperationBinder(store, schemas, artifacts)
    for operations in ALL_OPERATION_GROUPS:
        for adapter in installer.bind(operations).adapters:
            service.register(adapter)
    try:
        yield service
    finally:
        store.close()


def test_catalog_matches_installed_operations(service: OperationService) -> None:
    catalog_ids = {d.operation_id for d in service.catalog().operations}
    expected = _all_operation_ids()
    assert catalog_ids == expected, (
        f"missing from catalog: {sorted(expected - catalog_ids)}\n"
        f"extra in catalog: {sorted(catalog_ids - expected)}"
    )
    by_id: dict[str, OperationDescriptor] = {
        d.operation_id: d for d in service.catalog().operations
    }
    for operations in ALL_OPERATION_GROUPS:
        for operation in operations:
            desc = by_id[operation.operation_id]
            assert desc.version == operation.version
            assert desc.title == operation.title
            assert desc.description == operation.description
            assert desc.provider == "built-in"
            assert desc.tags == operation.tags
