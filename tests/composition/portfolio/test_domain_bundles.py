"""Portfolio-level tests for the DomainBundle architecture."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from jacobian.artifacts import ArtifactService
from jacobian.capability_service import CapabilityService
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
)
from jacobian.domains.arithmetic import build_arithmetic_bundle
from jacobian.domains.combinatorics import build_combinatorics_bundle
from jacobian.domains.finite_sets import build_finite_set_bundle
from jacobian.domains.number_theory import build_number_theory_bundle
from jacobian.domains.sequences import build_sequence_bundle
from jacobian.operation_installation import OperationInstaller
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository

ALL_BUNDLES = (
    build_arithmetic_bundle(),
    build_combinatorics_bundle(),
    build_finite_set_bundle(),
    build_number_theory_bundle(),
    build_sequence_bundle(),
)


def _all_operation_ids() -> set[str]:
    ids: set[str] = set()
    for bundle in ALL_BUNDLES:
        for operation in bundle.capabilities:
            ids.add(operation.spec.operation_id)
    return ids


def test_installed_bundles_expose_operations() -> None:
    actual = _all_operation_ids()
    assert actual, "expected at least one DomainBundle operation"
    assert len(actual) == sum(len(bundle.capabilities) for bundle in ALL_BUNDLES)


def test_unique_ids_within_each_bundle() -> None:
    for bundle in ALL_BUNDLES:
        ids = [op.spec.operation_id for op in bundle.capabilities]
        assert len(ids) == len(set(ids)), (
            f"{bundle.domain_id}: duplicates {[i for i in ids if ids.count(i) > 1]}"
        )


def test_no_id_in_two_bundles() -> None:
    seen: dict[str, str] = {}
    for bundle in ALL_BUNDLES:
        for operation in bundle.capabilities:
            cap_id = operation.spec.operation_id
            assert cap_id not in seen, (
                f"{cap_id!r} in both {seen[cap_id]!r} and {bundle.domain_id!r}"
            )
            seen[cap_id] = bundle.domain_id


def test_unique_domain_ids() -> None:
    domain_ids = [b.domain_id for b in ALL_BUNDLES]
    assert len(domain_ids) == len(set(domain_ids)), f"duplicates: {domain_ids}"


@pytest.fixture
def service(tmp_path: Path) -> Iterator[CapabilityService]:
    store = ArtifactRepository(tmp_path / "state")
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    service = CapabilityService(store)
    installer = OperationInstaller(store, schemas, artifacts)
    for bundle in ALL_BUNDLES:
        for adapter in installer.install(bundle).adapters:
            service.register(adapter)
    try:
        yield service
    finally:
        store.close()


def test_catalog_matches_installed_operations(service: CapabilityService) -> None:
    catalog_ids = {d.capability_id for d in service.catalog().capabilities}
    expected = _all_operation_ids()
    assert catalog_ids == expected, (
        f"missing from catalog: {sorted(expected - catalog_ids)}\n"
        f"extra in catalog: {sorted(catalog_ids - expected)}"
    )
    by_id: dict[str, CapabilityDescriptor] = {
        d.capability_id: d for d in service.catalog().capabilities
    }
    for bundle in ALL_BUNDLES:
        for operation in bundle.capabilities:
            desc = by_id[operation.spec.operation_id]
            assert desc.version == operation.spec.version
            assert desc.title == operation.spec.title
            assert desc.description == operation.spec.description
            assert desc.provider == bundle.provider_runtime.provider
            assert desc.tags == operation.spec.tags
