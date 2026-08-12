"""Unit tests for FixtureDef-oriented resource contracts."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from tests.support import resource_closure_plugin
from tests.support.resource_closure_plugin import (
    _authority_errors,
    _contract_for_name,
    _node_inventory,
    _resolved_contracts,
)
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


def _fixture_with_identity(
    module: str,
    *,
    resources: set[ResourceKind],
) -> Callable[[], None]:
    def shared_fixture_name() -> None:
        return None

    shared_fixture_name.__module__ = module
    shared_fixture_name.__qualname__ = "shared_fixture_name"
    return resource_fixture(
        resources=resources,
        isolation=IsolationClass.PRIVATE_MUTABLE,
    )(shared_fixture_name)


def test_parent_and_child_overrides_keep_distinct_function_identities() -> None:
    parent = _fixture_with_identity(
        "tests.parent.conftest", resources={ResourceKind.SQLITE}
    )
    child = _fixture_with_identity("tests.child.conftest", resources={ResourceKind.MCP})
    item: Any = SimpleNamespace(
        fixturenames=("shared_fixture_name",),
        _fixtureinfo=SimpleNamespace(
            name2fixturedefs={
                "shared_fixture_name": (
                    SimpleNamespace(func=parent),
                    SimpleNamespace(func=child),
                )
            }
        ),
    )

    contracts = _resolved_contracts(item)

    assert [contract.module for contract in contracts] == [
        "tests.parent.conftest",
        "tests.child.conftest",
    ]
    assert [contract.resources for contract in contracts] == [
        frozenset({ResourceKind.SQLITE}),
        frozenset({ResourceKind.MCP}),
    ]


def test_fixturedef_function_identity_wins_over_name_registration() -> None:
    fixturedef_func = _fixture_with_identity(
        "tests.identity.conftest", resources={ResourceKind.SQLITE}
    )
    latest_same_name = _fixture_with_identity(
        "tests.name_fallback.conftest", resources={ResourceKind.PROCESS_GROUP}
    )
    item: Any = SimpleNamespace(
        _fixtureinfo=SimpleNamespace(
            name2fixturedefs={
                "shared_fixture_name": (SimpleNamespace(func=fixturedef_func),)
            }
        )
    )

    assert resource_contract("shared_fixture_name") is resource_contract_for_function(
        latest_same_name
    )
    assert _contract_for_name(item, "shared_fixture_name") is (
        resource_contract_for_function(fixturedef_func)
    )
    assert (
        _contract_for_name(SimpleNamespace(_fixtureinfo=None), "shared_fixture_name")
        is None
    )


def test_node_inventory_reports_semantic_isolation_and_teardown_owner() -> None:
    @resource_fixture(
        resources={ResourceKind.SQLITE},
        isolation=IsolationClass.LIFECYCLE_OWNER,
        share_scope="module",
    )
    def inventory_fixture() -> None:
        return None

    contract = resource_contract_for_function(inventory_fixture)
    assert contract is not None
    item: Any = SimpleNamespace(
        nodeid="tests/composition/test_inventory.py::test_case",
        path="tests/composition/test_inventory.py",
        get_closest_marker=lambda name: None,
    )

    inventory = _node_inventory(item, [contract])

    assert inventory["semantic_owner"] == "composition"
    assert inventory["isolation"] == ["lifecycle-owner"]
    assert inventory["teardown_owner"] == ["module"]


def test_composition_authority_requires_marker_even_with_source_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resource_closure_plugin,
        "_module_source",
        lambda item: "verification_record_uri is not None",
    )
    item: Any = SimpleNamespace(nodeid="tests/composition/test_authority.py::test_case")

    errors = _authority_errors(
        item,
        admission=None,
        relative="tests/composition/test_authority.py",
    )

    assert errors
    assert "composition_admission('AUTHORITY')" in errors[0]
