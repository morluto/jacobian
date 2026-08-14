from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationDiscoveryRequest,
    OperationInputKind,
)
from jacobian.operation_catalog import (
    CompiledCatalogEntry,
    OperationCatalog,
    OperationCatalogError,
    OperationCatalogStore,
)
from jacobian.operation_visibility import OperationVisibilityPolicy
from jacobian.registry import CheckerRegistry
from jacobian.serving_catalog import ServingCatalog
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
        declaration_module="jacobian.domains.synthetic.domain_declarations",
        declaration_digest="sha256:" + "a" * 64,
    )


def _store(root: Path) -> OperationCatalogStore:
    with ArtifactRepository(root):
        pass
    return OperationCatalogStore(root / "metadata.sqlite3")


def _commit(store: OperationCatalogStore) -> int:
    result = store.commit(
        package_version="0.13.0",
        checker_binding_digest="sha256:" + "c" * 64,
        entries=(
            _entry("integer.gcd.compute", "Greatest common divisor"),
            _entry("matrix.rank.compute", "Matrix rank"),
        ),
        checker_bindings={},
    )
    return result.revision


def _commit_with_checker(store: OperationCatalogStore) -> tuple[str, str]:
    with ArtifactRepository(store.database_path.parent) as repository:
        checker = CheckerRegistry(repository).authorize(
            name="reject-all-v1",
            entrypoint="jacobian_checkers.reject:check",
            evidence_kind="WITNESS",
            format_id="example.witness",
            format_version="1",
            claim_schema_uris=("artifact://sha256/" + "a" * 64,),
            semantics_uris=("artifact://sha256/" + "a" * 64,),
            candidate_schema_uris=("artifact://sha256/" + "a" * 64,),
        )
    store.commit(
        package_version="0.13.0",
        checker_binding_digest="sha256:" + "c" * 64,
        entries=(
            _entry("integer.gcd.compute", "Greatest common divisor"),
            _entry("matrix.rank.compute", "Matrix rank"),
        ),
        checker_bindings={
            "matrix.rank.compute": (
                (checker.checker_id, checker.implementation_digest),
                (checker.checker_id, checker.implementation_digest),
            )
        },
    )
    return checker.checker_id, checker.implementation_digest


def test_catalog_search_uses_cards_when_descriptors_are_not_materializable(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _commit(store)
    catalog = OperationCatalog(
        tmp_path / "metadata.sqlite3",
        OperationVisibilityPolicy(),
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


def test_catalog_search_cards_preserve_typed_input_compatibility(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    artifact_type = "artifact://sha256/" + "d" * 64
    descriptor = _descriptor("graph.consume.typed", "Typed graph consumer")
    descriptor = descriptor.model_copy(
        update={
            "accepted_input_kinds": (OperationInputKind.TYPED_ARTIFACT,),
            "accepted_artifact_types": (artifact_type,),
        }
    )
    formal_descriptor = _descriptor(
        "lean.statement.compare",
        "Formal statement comparison",
    ).model_copy(
        update={
            "accepted_input_kinds": (OperationInputKind.FORMAL_PROPOSITION,),
        }
    )
    store.commit(
        package_version="0.13.0",
        checker_binding_digest="sha256:" + "c" * 64,
        entries=(
            CompiledCatalogEntry(
                descriptor=descriptor,
                declaration_module=("jacobian.domains.synthetic.domain_declarations"),
                declaration_digest="sha256:" + "a" * 64,
            ),
            CompiledCatalogEntry(
                descriptor=formal_descriptor,
                declaration_module=("jacobian.lean_frontend.statement"),
                declaration_digest="sha256:" + "e" * 64,
            ),
        ),
        checker_bindings={},
    )
    catalog = OperationCatalog(
        tmp_path / "metadata.sqlite3",
        OperationVisibilityPolicy(),
        expected_package_version="0.13.0",
    )

    result = catalog.search(
        OperationDiscoveryRequest(
            query="typed graph",
            input_kind=OperationInputKind.TYPED_ARTIFACT,
            artifact_type=artifact_type,
        )
    )

    assert len(result.matches) == 1
    assert result.matches[0].applicability == "NEEDS_MORE_TYPED_REQUIREMENTS"

    formal_result = catalog.search(
        OperationDiscoveryRequest(
            query="formal statement",
            input_kind=OperationInputKind.FORMAL_PROPOSITION,
        )
    )
    assert len(formal_result.matches) == 1
    assert formal_result.matches[0].applicability == "NEEDS_MORE_TYPED_REQUIREMENTS"


def test_catalog_inspection_reads_one_active_indexed_entry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    revision = _commit(store)
    catalog = OperationCatalog(
        tmp_path / "metadata.sqlite3",
        OperationVisibilityPolicy(),
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


def test_catalog_reads_one_exact_declaration_locator(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _commit(store)
    catalog = OperationCatalog(
        tmp_path / "metadata.sqlite3",
        OperationVisibilityPolicy(),
        expected_package_version="0.13.0",
    )

    record = catalog.declaration_record("matrix.rank.compute")

    assert record is not None
    assert record.module == "jacobian.domains.synthetic.domain_declarations"
    assert record.declaration_digest == "sha256:" + "a" * 64
    assert catalog.declaration_record("missing.operation") is None


def test_serving_catalog_rejects_packaged_builtin_mirrors(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _commit(store)

    with pytest.raises(
        OperationCatalogError, match="operation catalog overlay is stale"
    ):
        ServingCatalog.open(
            tmp_path / "metadata.sqlite3",
            OperationVisibilityPolicy(),
            expected_package_version="0.13.0",
        )


def test_serving_catalog_hides_omitted_packaged_ids(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.commit(
        package_version="0.13.0",
        checker_binding_digest="sha256:" + "c" * 64,
        entries=(_entry("synthetic.overlay.only", "Overlay only"),),
        checker_bindings={},
        omitted_operations=("lean.check", "sat.model.find"),
    )

    catalog = ServingCatalog.open(
        tmp_path / "metadata.sqlite3",
        OperationVisibilityPolicy(),
        expected_package_version="0.13.0",
    )

    assert catalog.inspect("lean.check") is None
    assert catalog.inspect("sat.model.find") is None
    assert catalog.declaration_record("lean.check") is None
    assert catalog.inspect("graph.construct.explicit") is not None
    assert catalog.inspect("sat.cnf.materialize") is not None
    assert catalog.inspect("synthetic.overlay.only") is not None
    snapshot_ids = {item.operation_id for item in catalog.snapshot().operations}
    assert "lean.check" not in snapshot_ids
    assert "sat.model.find" not in snapshot_ids
    assert "graph.construct.explicit" in snapshot_ids
    assert "synthetic.overlay.only" in snapshot_ids


def test_catalog_loads_the_selected_checker_binding_index(tmp_path: Path) -> None:
    store = _store(tmp_path)
    checker_id, implementation_digest = _commit_with_checker(store)
    catalog = OperationCatalog(
        tmp_path / "metadata.sqlite3",
        OperationVisibilityPolicy(),
        expected_package_version="0.13.0",
    )

    binding = catalog.checker_binding("matrix.rank.compute")

    assert binding is not None
    assert binding.binding_index == 0
    assert binding.checker_id == checker_id
    assert binding.manifest_digest == implementation_digest
    bindings = catalog.checker_bindings("matrix.rank.compute")
    assert tuple(item.binding_index for item in bindings) == (0, 1)
    assert tuple(item.checker_id for item in bindings) == (checker_id, checker_id)
    assert catalog.checker_binding("missing.operation") is None
    assert catalog.checker_bindings("missing.operation") == ()


def test_failed_catalog_commit_leaves_previous_revision_active(tmp_path: Path) -> None:
    store = _store(tmp_path)
    revision = _commit(store)

    with pytest.raises(sqlite3.IntegrityError):
        store.commit(
            package_version="0.13.0",
            checker_binding_digest="sha256:" + "e" * 64,
            entries=(_entry("integer.gcd.compute", "Changed GCD"),),
            checker_bindings={
                "integer.gcd.compute": (
                    (
                        "checker://sha256/" + "f" * 64,
                        "sha256:" + "1" * 64,
                    ),
                )
            },
        )

    catalog = OperationCatalog(
        tmp_path / "metadata.sqlite3",
        OperationVisibilityPolicy(),
        expected_package_version="0.13.0",
    )
    assert catalog.header.revision == revision
    assert catalog.inspect("integer.gcd.compute").title == "Greatest common divisor"  # type: ignore[union-attr]


def test_catalog_requires_init_or_update_with_exact_commands(tmp_path: Path) -> None:
    with pytest.raises(
        OperationCatalogError,
        match=r"STATE_INITIALIZATION_REQUIRED: run `jacobian init`",
    ):
        OperationCatalog(
            tmp_path / "missing" / "metadata.sqlite3",
            OperationVisibilityPolicy(),
            expected_package_version="0.13.0",
        )

    store = _store(tmp_path)
    with pytest.raises(OperationCatalogError, match="jacobian init"):
        OperationCatalog(
            tmp_path / "metadata.sqlite3",
            OperationVisibilityPolicy(),
            expected_package_version="0.13.0",
        )

    _commit(store)
    with pytest.raises(OperationCatalogError, match="jacobian update"):
        OperationCatalog(
            tmp_path / "metadata.sqlite3",
            OperationVisibilityPolicy(),
            expected_package_version="0.13.1",
        )
