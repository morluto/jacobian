"""Complete-runtime and service fixtures owned by composition tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.support.complete_runtime_fixtures import (
    attached_complete_runtime,
    attached_complete_runtime_read_only,
    authorized_complete_runtime,
    authorized_complete_runtime_read_only,
    authorized_portfolio_template,
    complete_portfolio_template,
    fresh_complete_runtime,
)
from tests.support.services import DomainTestServices, open_domain_services

# Directory-scoped fixture registration (not pytest_plugins).
__all__ = (
    "attached_complete_runtime",
    "attached_complete_runtime_read_only",
    "authorized_complete_runtime",
    "authorized_complete_runtime_read_only",
    "authorized_portfolio_template",
    "complete_portfolio_template",
    "fresh_complete_runtime",
    "operation_core_services",
)


@pytest.fixture
def operation_core_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Open production core/application seams for service-level composition tests."""

    with open_domain_services(tmp_path / "state") as services:
        yield services
