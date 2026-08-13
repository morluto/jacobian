from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.core_operation_harnesses import (
    UniversalAlgebraTestServices,
    open_universal_algebra_services,
)


@pytest.fixture
def universal_algebra_services(
    tmp_path: Path,
) -> Iterator[UniversalAlgebraTestServices]:
    with open_universal_algebra_services(tmp_path / "state") as services:
        yield services


@pytest.fixture
def unauthorized_universal_algebra_services(
    tmp_path: Path,
) -> Iterator[UniversalAlgebraTestServices]:
    with open_universal_algebra_services(
        tmp_path / "state",
        authorize_checker=False,
    ) as services:
        yield services
