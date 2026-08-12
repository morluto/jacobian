"""Portfolio-level tests for the DomainBundle architecture."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.resource_contracts import (
    IsolationClass,
    ResourceKind,
    resource_fixture,
)

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
from jacobian.operations import DomainBundle
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository

ALL_BUNDLES = (
    build_arithmetic_bundle(),
    build_combinatorics_bundle(),
    build_finite_set_bundle(),
    build_number_theory_bundle(),
    build_sequence_bundle(),
)


def _operations(bundle: DomainBundle) -> tuple[object, ...]:
    return tuple(bundle.capabilities)


def _all_operation_ids() -> set[str]:
    ids: set[str] = set()
    for bundle in ALL_BUNDLES:
        for operation in _operations(bundle):
            ids.add(operation.spec.operation_id)
    return ids


@pytest.mark.parametrize(
    "bundle",
    ALL_BUNDLES,
    ids=[bundle.domain_id for bundle in ALL_BUNDLES],
)
def test_bundle_exposes_unique_operation_ids(bundle: DomainBundle) -> None:
    ids = [op.spec.operation_id for op in _operations(bundle)]
    assert ids, f"{bundle.domain_id}: expected at least one operation"
    assert len(ids) == len(set(ids)), (
        f"{bundle.domain_id}: duplicates {[i for i in ids if ids.count(i) > 1]}"
    )


def test_no_id_in_two_bundles() -> None:
    seen: dict[str, str] = {}
    for bundle in ALL_BUNDLES:
        for operation in _operations(bundle):
            cap_id = operation.spec.operation_id
            assert cap_id not in seen, (
                f"{cap_id!r} in both {seen[cap_id]!r} and {bundle.domain_id!r}"
            )
            seen[cap_id] = bundle.domain_id


def test_unique_domain_ids() -> None:
    domain_ids = [b.domain_id for b in ALL_BUNDLES]
    assert len(domain_ids) == len(set(domain_ids)), f"duplicates: {domain_ids}"


def test_installed_bundles_expose_operations() -> None:
    actual = _all_operation_ids()
    assert actual, "expected at least one DomainBundle operation"
    assert len(actual) == sum(len(_operations(bundle)) for bundle in ALL_BUNDLES)


@pytest.fixture
@resource_fixture(
    resources={ResourceKind.SQLITE},
    isolation=IsolationClass.LIFECYCLE_OWNER,
    profile_key="portfolio-domain-bundles-v1",
    setup_affinity="sqlite",
)
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


@pytest.mark.parametrize(
    "bundle",
    ALL_BUNDLES,
    ids=[bundle.domain_id for bundle in ALL_BUNDLES],
)
def test_catalog_matches_bundle_operations(
    service: CapabilityService,
    bundle: DomainBundle,
) -> None:
    by_id: dict[str, CapabilityDescriptor] = {
        d.capability_id: d for d in service.catalog().capabilities
    }
    for operation in _operations(bundle):
        desc = by_id[operation.spec.operation_id]
        assert desc.version == operation.spec.version
        assert desc.title == operation.spec.title
        assert desc.description == operation.spec.description
        assert desc.provider == bundle.provider_runtime.provider
        assert desc.tags == operation.spec.tags


def test_catalog_contains_exactly_installed_operations(
    service: CapabilityService,
) -> None:
    catalog_ids = {d.capability_id for d in service.catalog().capabilities}
    expected = _all_operation_ids()
    assert catalog_ids == expected, (
        f"missing from catalog: {sorted(expected - catalog_ids)}\n"
        f"extra in catalog: {sorted(catalog_ids - expected)}"
    )
