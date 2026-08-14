from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.domains.topology import topology_operations


@pytest.fixture
def topology_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install topology and its exact checkers without a portfolio."""

    with open_exact_domain_services(
        tmp_path / "state",
        topology_operations(),
    ) as services:
        yield services
