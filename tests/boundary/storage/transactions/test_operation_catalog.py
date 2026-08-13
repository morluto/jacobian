from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiscoveryRequest,
)
from jacobian.operation_catalog import (
    CompiledCatalogEntry,
    OperationCatalog,
    OperationCatalogError,
    OperationCatalogStore,
)
from jacobian.operation_service import OperationPolicy
from jacobian.storage.repository import ArtifactRepository


def _descriptor(operation_id: str, title: str) -> OperationDescriptor:
    return OperationDescriptor(
        operation_id=operation_id,
        version="1",
        title=title,
        description=f"Compute {title.casefold()} exactly.",
        provider="built-in",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        read_only=True,
        tags=("exact",),
    )


def _entry(operation_id: str, title: str) -> CompiledCatalogEntry:
    return CompiledCatalogEntry(
        descriptor=_descriptor(operation_id, title),
        bundle_module="jacobian.domains.synthetic.bundle",
        declaration_digest="sha256:" + "a" * 64,
    )


def _store(root: Path) -> OperationCatalogStore:
    with ArtifactRepository(root):
        pass
    return OperationCatalogStore(root / "metadata.sqlite3")


def _commit(store: OperationCatalogStore) -> int:
    result = store.commit(
        package_version="0.13.0",
        provider_inventory_digest="sha256:" + "b" * 64,
        checker_binding_digest="sha256:" + "c" * 64,
        entries=(
            _entry("integer.gcd.compute", "Greatest common divisor"),
            _entry("matrix.rank.compute", "Matrix rank"),
        ),
        checker_bindings={},
    )
    return result.revision


def test_catalog_search_uses_cards_when_descriptors_are_not_materializable(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _commit(store)
    catalog = OperationCatalog(
        tmp_path / "metadata.sqlite3",
        OperationPolicy(),
        expected_package_version="0.13.0",
    )
    with sqlite3.connect(tmp_path / "metadata.sqlite3") as connection:
        connection.execute(
            "UPDATE operation_catalog_entries SET descriptor_json = 'not-json'"
        )

    result = catalog.search(OperationDiscoveryRequest(query="greatest divisor"))

    assert tuple(match.operation_id for match in result.matches) == (
        "integer.gcd.compute",
    )
    with pytest.raises(ValueError):
        catalog.inspect("integer.gcd.compute")


def test_catalog_inspection_reads_one_active_indexed_entry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    revision = _commit(store)
    catalog = OperationCatalog(
        tmp_path / "metadata.sqlite3",
        OperationPolicy(),
        expected_package_version="0.13.0",
    )

    descriptor = catalog.inspect("matrix.rank.compute")

    assert descriptor is not None
    assert descriptor.title == "Matrix rank"
    with sqlite3.connect(tmp_path / "metadata.sqlite3") as connection:
        plan = tuple(
            row[3]
            for row in connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT descriptor_json FROM operation_catalog_entries
                WHERE snapshot_revision = ? AND operation_id = ?
                """,
                (revision, "matrix.rank.compute"),
            )
        )
    assert any("PRIMARY KEY" in detail or "INDEX" in detail for detail in plan)


def test_failed_catalog_commit_leaves_previous_revision_active(tmp_path: Path) -> None:
    store = _store(tmp_path)
    revision = _commit(store)

    with pytest.raises(sqlite3.IntegrityError):
        store.commit(
            package_version="0.13.0",
            provider_inventory_digest="sha256:" + "d" * 64,
            checker_binding_digest="sha256:" + "e" * 64,
            entries=(_entry("integer.gcd.compute", "Changed GCD"),),
            checker_bindings={
                "integer.gcd.compute": (
                    "checker://sha256/" + "f" * 64,
                    "sha256:" + "1" * 64,
                )
            },
        )

    catalog = OperationCatalog(
        tmp_path / "metadata.sqlite3",
        OperationPolicy(),
        expected_package_version="0.13.0",
    )
    assert catalog.header.revision == revision
    assert catalog.inspect("integer.gcd.compute").title == "Greatest common divisor"  # type: ignore[union-attr]


def test_catalog_requires_init_or_update_with_exact_commands(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(OperationCatalogError, match="jacobian init"):
        OperationCatalog(
            tmp_path / "metadata.sqlite3",
            OperationPolicy(),
            expected_package_version="0.13.0",
        )

    _commit(store)
    with pytest.raises(OperationCatalogError, match="jacobian update"):
        OperationCatalog(
            tmp_path / "metadata.sqlite3",
            OperationPolicy(),
            expected_package_version="0.13.1",
        )
