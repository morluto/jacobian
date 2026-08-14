from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.domains.matrix_lattice import matrix_operations


@pytest.fixture(scope="module")
def matrix_services(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[DomainTestServices]:
    """Install matrix operations and their exact checkers without a portfolio."""

    with open_exact_domain_services(
        tmp_path_factory.mktemp("matrix-domain") / "state",
        matrix_operations(),
    ) as services:
        yield services
