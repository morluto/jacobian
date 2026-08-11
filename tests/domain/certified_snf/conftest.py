from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.domains.certified_snf import build_certified_snf_bundle


@pytest.fixture
def certified_snf_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install certified Smith normal form and its checker without a portfolio."""

    with open_exact_domain_services(
        tmp_path / "state",
        build_certified_snf_bundle(),
    ) as services:
        yield services
