from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.domains.matrix_lattice import build_matrix_bundle


@pytest.fixture
def matrix_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install matrix operations and their exact checkers without a portfolio."""

    with open_exact_domain_services(
        tmp_path / "state",
        build_matrix_bundle(),
    ) as services:
        yield services
