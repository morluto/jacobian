"""Collection-time resource-closure reporting for the test architecture overhaul.

Report-only plugin: records requested fixture names that have registered
resource contracts. Enforcement ratchets land after profiles stabilize.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.support.resource_contracts import resource_contract


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "jacobian_resources(name): declare an explicit resource profile override",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    del session, config
    for item in items:
        fixturenames = getattr(item, "fixturenames", ())
        contracts = [
            resource_contract(name)
            for name in fixturenames
            if resource_contract(name) is not None
        ]
        if contracts:
            item.user_properties.append(
                (
                    "jacobian_resources",
                    sorted(
                        {
                            resource.value
                            for contract in contracts
                            if contract is not None
                            for resource in contract.resources
                        }
                    ),
                )
            )


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]) -> None:
    del item, call
