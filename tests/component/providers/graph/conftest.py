"""Focused graph-provider service graph."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.domains.graph_optimization.bundle import (
    build_graph_optimization_bundle,
)


@pytest.fixture
def graph_optimization_services(
    tmp_path: Path,
) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state",
        build_graph_optimization_bundle(),
    ) as services:
        yield services
