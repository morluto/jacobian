"""Collection-time enforcement of typed fixture resource contracts.

Rejects complete-runtime fixtures outside owning semantic/boundary paths.
Composition authority requires an explicit per-node marker; other lanes retain
the temporary module-source authority fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tools.test_plan.authority_signals import has_verify_authority_signal
from tools.test_plan.runtime_owners import allows_complete_runtime_fixture

from tests.support.resource_contracts import (
    IsolationClass,
    ResourceFixtureContract,
    ResourceKind,
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


def _contracts_for_name(
    item: pytest.Item, name: str
) -> list[tuple[tuple[str, str], ResourceFixtureContract]]:
    fixtureinfo = getattr(item, "_fixtureinfo", None)
    name2fixturedefs = getattr(fixtureinfo, "name2fixturedefs", None)
    if not isinstance(name2fixturedefs, dict):
        return []
    contracts: list[tuple[tuple[str, str], ResourceFixtureContract]] = []
    for fixturedef in name2fixturedefs.get(name) or ():
        func = getattr(fixturedef, "func", None)
        if not callable(func):
            continue
        contract = resource_contract_for_function(func)
        if contract is None:
            continue
        identity = (
            getattr(func, "__module__", "") or "",
            getattr(func, "__qualname__", "") or getattr(func, "__name__", ""),
        )
        contracts.append((identity, contract))
    return contracts


def _contract_for_name(item: pytest.Item, name: str) -> ResourceFixtureContract | None:
    """Resolve the active contract from the nearest FixtureDef function."""

    contracts = _contracts_for_name(item, name)
    return contracts[-1][1] if contracts else None


def _resolved_contracts(item: pytest.Item) -> list[ResourceFixtureContract]:
    """Resolve contracts from FixtureDef functions when available."""

    contracts: list[ResourceFixtureContract] = []
    seen: set[tuple[str, str]] = set()
    for name in getattr(item, "fixturenames", ()):
        for identity, contract in _contracts_for_name(item, name):
            if identity in seen:
                continue
            seen.add(identity)
            contracts.append(contract)
    return contracts


def _semantic_owner(relative: str) -> str | None:
    for owner in ("unit", "component", "domain", "composition", "e2e", "boundary"):
        prefix = f"tests/{owner}"
        if relative == prefix or relative.startswith(prefix + "/"):
            return owner
    return None


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
    isolation = sorted({contract.isolation.value for contract in contracts})
    teardown_owner = sorted(
        {
            contract.share_scope
            for contract in contracts
            if contract.isolation is IsolationClass.LIFECYCLE_OWNER
        }
    )
    relative = _relative_path(item)
    return {
        "nodeid": item.nodeid,
        "path": relative,
        "semantic_owner": _semantic_owner(relative),
        "resources": resources,
        "isolation": isolation,
        "teardown_owner": teardown_owner,
        "setup_affinity": affinities,
        "profile_keys": profiles,
        "composition_admission": _marker_value(item, _COMPOSITION_ADMISSION_MARKER),
        "fixtures": [contract.name for contract in contracts],
    }


def _authority_errors(
    item: pytest.Item,
    *,
    admission: str | None,
    relative: str,
) -> list[str]:
    if admission in _AUTHORITY_ADMISSION:
        return []
    if admission is not None:
        return [
            f"{item.nodeid}: authorized_complete_runtime requires "
            "composition_admission('AUTHORITY') when a marker is present"
        ]
    if relative == "tests/composition" or relative.startswith("tests/composition/"):
        return [
            f"{item.nodeid}: authorized_complete_runtime requires "
            "composition_admission('AUTHORITY') under tests/composition/"
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
        errors.extend(_authority_errors(item, admission=admission, relative=relative))
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
