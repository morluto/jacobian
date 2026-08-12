"""Collection-time enforcement of typed fixture resource contracts.

Rejects complete-runtime fixtures outside owning semantic/boundary paths and
rejects authorized-checker hydration without a verify/authority signal.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.test_architecture.authority_signals import has_verify_authority_signal
from tools.test_architecture.runtime_owners import allows_complete_runtime_fixture

from tests.support.resource_contracts import ResourceKind, resource_contract


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


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    del session, config
    errors: list[str] = []
    for item in items:
        relative = _relative_path(item)
        fixturenames = getattr(item, "fixturenames", ())
        contracts = [
            contract
            for name in fixturenames
            if (contract := resource_contract(name)) is not None
        ]
        if not contracts:
            continue
        resources = {
            resource for contract in contracts for resource in contract.resources
        }
        item.user_properties.append(
            ("jacobian_resources", sorted(resource.value for resource in resources))
        )
        if (
            ResourceKind.COMPLETE_RUNTIME in resources
            and not allows_complete_runtime_fixture(relative)
        ):
            errors.append(
                f"{item.nodeid}: complete-runtime fixtures are not permitted "
                f"under {relative}; use open_domain_services or move the test"
            )
        if ResourceKind.AUTHORIZED_CHECKERS in resources:
            source = _module_source(item)
            if not has_verify_authority_signal(source):
                errors.append(
                    f"{item.nodeid}: authorized_complete_runtime requires a "
                    "verify/authority assertion in the module source"
                )
    if errors:
        raise pytest.UsageError(
            "jacobian resource-closure policy violations:\n" + "\n".join(errors)
        )
