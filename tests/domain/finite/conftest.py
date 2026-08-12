from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.core_capability_harnesses import (
    FiniteCoverageTestServices,
    open_finite_coverage_services,
)
from tests.support.resource_contracts import (
    IsolationClass,
    ResourceKind,
    resource_fixture,
)


@pytest.fixture
@resource_fixture(
    resources={ResourceKind.SQLITE},
    isolation=IsolationClass.LIFECYCLE_OWNER,
    profile_key="finite-coverage-v1",
    setup_affinity="sqlite",
)
def finite_coverage_services(
    tmp_path: Path,
) -> Iterator[FiniteCoverageTestServices]:
    with open_finite_coverage_services(tmp_path / "state") as services:
        yield services


@pytest.fixture
@resource_fixture(
    resources={ResourceKind.SQLITE},
    isolation=IsolationClass.LIFECYCLE_OWNER,
    profile_key="finite-coverage-unauthorized-v1",
    setup_affinity="sqlite",
)
def unauthorized_finite_coverage_services(
    tmp_path: Path,
) -> Iterator[FiniteCoverageTestServices]:
    with open_finite_coverage_services(
        tmp_path / "state",
        authorize_checker=False,
    ) as services:
        yield services
