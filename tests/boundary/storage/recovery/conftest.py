from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from search_orchestration_support import open_search_services
from tests.support.services import (
    DomainTestServices,
    ReferenceTestServices,
    open_reference_services,
)


@pytest.fixture
def search_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Provide a private production search graph without portfolio setup."""

    with open_search_services(tmp_path / "state") as services:
        yield services


@pytest.fixture
def matrix_reference_services(tmp_path: Path) -> Iterator[ReferenceTestServices]:
    """Provide a private matrix enumeration graph."""

    with open_reference_services(tmp_path / "state", "matrices") as services:
        yield services


@pytest.fixture
def graph_reference_services(tmp_path: Path) -> Iterator[ReferenceTestServices]:
    """Provide a private graph enumeration graph."""

    with open_reference_services(tmp_path / "state", "graph_paths") as services:
        yield services
