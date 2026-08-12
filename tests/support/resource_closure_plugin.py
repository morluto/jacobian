"""Collection-time enforcement of typed fixture resource contracts.

Rejects complete-runtime fixtures outside owning semantic/boundary paths and
rejects authorized-checker hydration without a per-node or module verify signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tools.test_plan.authority_signals import has_verify_authority_signal
from tools.test_plan.runtime_owners import allows_complete_runtime_fixture

from tests.support.resource_contracts import (
    ResourceFixtureContract,
    ResourceKind,
    resource_contract,
    resource_contract_for_function,
)

_COMPOSITION_ADMISSION_MARKER = "composition_admission"
_COMPOSITION_ADMISSION_VALUES = frozenset(
    {"AUTHORITY", "WIRING", "LIFECYCLE", "DISCOVERY", "REFERENCE"}
)
_AUTHORITY_ADMISSION = frozenset({"AUTHORITY"})


def _relative_path(item: pytest.Item) -> str:
    path = Path(str(item.path))
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _module_source(item: pytest.Item) -> str:
    path = Path(str(item.path))
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _marker_value(item: pytest.Item, name: str) -> str | None:
    marker = item.get_closest_marker(name)
    if marker is None or not marker.args:
        return None
    value = marker.args[0]
    return value if isinstance(value, str) else None


def _contract_for_name(item: pytest.Item, name: str) -> ResourceFixtureContract | None:
    fixtureinfo = getattr(item, "_fixtureinfo", None)
    name2fixturedefs = getattr(fixtureinfo, "name2fixturedefs", None)
    if isinstance(name2fixturedefs, dict):
        for fixturedef in name2fixturedefs.get(name) or ():
            func = getattr(fixturedef, "func", None)
            if callable(func):
                contract = resource_contract_for_function(func)
                if contract is not None:
                    return contract
    return resource_contract(name)


def _resolved_contracts(item: pytest.Item) -> list[ResourceFixtureContract]:
    """Resolve contracts from FixtureDef functions when available."""

    contracts: list[ResourceFixtureContract] = []
    seen: set[str] = set()
    for name in getattr(item, "fixturenames", ()):
        contract = _contract_for_name(item, name)
        if contract is None or contract.name in seen:
            continue
        seen.add(contract.name)
        contracts.append(contract)
    return contracts


def _node_inventory(
    item: pytest.Item, contracts: list[ResourceFixtureContract]
) -> dict[str, Any]:
    resources = sorted(
        {resource.value for contract in contracts for resource in contract.resources}
    )
    affinities = sorted(
        {contract.setup_affinity for contract in contracts if contract.setup_affinity}
    )
    profiles = sorted(
        {contract.profile_key for contract in contracts if contract.profile_key}
    )
    return {
        "nodeid": item.nodeid,
        "path": _relative_path(item),
        "resources": resources,
        "setup_affinity": affinities,
        "profile_keys": profiles,
        "composition_admission": _marker_value(item, _COMPOSITION_ADMISSION_MARKER),
        "fixtures": [contract.name for contract in contracts],
    }


def _authority_errors(
    item: pytest.Item,
    *,
    admission: str | None,
) -> list[str]:
    if admission in _AUTHORITY_ADMISSION:
        return []
    if admission is not None:
        return [
            f"{item.nodeid}: authorized_complete_runtime requires "
            "composition_admission('AUTHORITY') when a marker is present"
        ]
    if has_verify_authority_signal(_module_source(item)):
        return []
    return [
        f"{item.nodeid}: authorized_complete_runtime requires "
        "composition_admission('AUTHORITY') or a verify/authority "
        "assertion in the module source"
    ]


def _item_errors(
    item: pytest.Item, contracts: list[ResourceFixtureContract]
) -> list[str]:
    errors: list[str] = []
    relative = _relative_path(item)
    resources = {resource for contract in contracts for resource in contract.resources}
    admission = _marker_value(item, _COMPOSITION_ADMISSION_MARKER)
    if admission is not None and admission not in _COMPOSITION_ADMISSION_VALUES:
        errors.append(
            f"{item.nodeid}: composition_admission={admission!r} is not one of "
            f"{sorted(_COMPOSITION_ADMISSION_VALUES)}"
        )
    if (
        ResourceKind.COMPLETE_RUNTIME in resources
        and not allows_complete_runtime_fixture(relative)
    ):
        path = " -> ".join([item.nodeid, *[contract.name for contract in contracts]])
        errors.append(
            f"{item.nodeid}: complete-runtime fixtures are not permitted "
            f"under {relative}; dependency path: {path}; use "
            "open_domain_services or move the test"
        )
    if ResourceKind.AUTHORIZED_CHECKERS in resources:
        errors.extend(_authority_errors(item, admission=admission))
    return errors


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "composition_admission(kind): per-node complete-runtime admission "
        "(AUTHORITY|WIRING|LIFECYCLE|DISCOVERY|REFERENCE)",
    )


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    del session
    errors: list[str] = []
    inventory: list[dict[str, Any]] = []
    for item in items:
        contracts = _resolved_contracts(item)
        if contracts:
            inventory.append(_node_inventory(item, contracts))
            item.user_properties.append(
                (
                    "jacobian_resources",
                    sorted(
                        resource.value
                        for contract in contracts
                        for resource in contract.resources
                    ),
                )
            )
        errors.extend(_item_errors(item, contracts))
    if inventory:
        config._jacobian_resource_inventory = inventory  # type: ignore[attr-defined]
    if errors:
        raise pytest.UsageError(
            "jacobian resource-closure policy violations:\n" + "\n".join(errors)
        )
