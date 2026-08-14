"""Focused graph-provider service graph."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.domains.graph_optimization.domain_declarations import (
    graph_optimization_operations,
)


@pytest.fixture
def graph_optimization_services(
    tmp_path: Path,
) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state",
        graph_optimization_operations(),
    ) as services:
        yield services
