from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.core_capability_harnesses import (
    FiniteCoverageTestServices,
    FinitePartitionTestServices,
    open_finite_coverage_services,
    open_finite_partition_services,
)


@pytest.fixture
def finite_coverage_services(
    tmp_path: Path,
) -> Iterator[FiniteCoverageTestServices]:
    with open_finite_coverage_services(tmp_path / "state") as services:
        yield services


@pytest.fixture
def finite_partition_services(
    tmp_path: Path,
) -> Iterator[FinitePartitionTestServices]:
    with open_finite_partition_services(tmp_path / "state") as services:
        yield services


@pytest.fixture
def unauthorized_finite_partition_services(
    tmp_path: Path,
) -> Iterator[FinitePartitionTestServices]:
    with open_finite_partition_services(
        tmp_path / "state",
        authorize_checker=False,
    ) as services:
        yield services
