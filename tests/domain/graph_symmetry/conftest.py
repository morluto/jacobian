from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.domains.graph_symmetry import build_graph_symmetry_bundle


@pytest.fixture
def graph_symmetry_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install graph symmetry and its exact checkers without a portfolio."""

    with open_exact_domain_services(
        tmp_path / "state",
        build_graph_symmetry_bundle(),
    ) as services:
        yield services
