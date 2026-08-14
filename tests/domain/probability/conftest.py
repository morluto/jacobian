from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.domains.probability import finite_probability_operations


@pytest.fixture
def probability_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install finite probability and its exact checkers without a portfolio."""

    with open_exact_domain_services(
        tmp_path / "state",
        finite_probability_operations(),
    ) as services:
        yield services
