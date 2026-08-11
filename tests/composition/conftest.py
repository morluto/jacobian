"""Complete-runtime and service fixtures owned by composition tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.support.services import DomainTestServices, open_domain_services


@pytest.fixture
def capability_core_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Open production core/application seams for service-level composition tests."""

    with open_domain_services(tmp_path / "state") as services:
        yield services
