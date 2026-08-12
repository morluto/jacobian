"""Unit tests for FixtureDef-oriented resource contracts."""

from __future__ import annotations

from tests.support.resource_contracts import (
    IsolationClass,
    ResourceKind,
    registered_resource_fixtures,
    resource_contract,
    resource_contract_for_function,
    resource_fixture,
)


@resource_fixture(
    resources={ResourceKind.SQLITE},
    isolation=IsolationClass.PRIVATE_MUTABLE,
    profile_key="unit-probe-v1",
)
def _probe_resource_fixture() -> None:
    return None


def test_resource_fixture_binds_function_identity() -> None:
    by_name = resource_contract("_probe_resource_fixture")
    by_func = resource_contract_for_function(_probe_resource_fixture)
    assert by_name is not None
    assert by_func is not None
    assert by_name is by_func
    assert ResourceKind.SQLITE in by_func.resources
    assert by_func.module == __name__
    assert "_probe_resource_fixture" in registered_resource_fixtures()
