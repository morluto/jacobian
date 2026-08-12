from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.resource_contracts import (
    IsolationClass,
    ResourceKind,
    resource_fixture,
)
from tests.support.services import DomainTestServices

from jacobian.domains.probability import build_finite_probability_bundle


@pytest.fixture
@resource_fixture(
    resources={ResourceKind.SQLITE},
    isolation=IsolationClass.LIFECYCLE_OWNER,
    profile_key="exact-domain-probability-v1",
    setup_affinity="sqlite",
)
def probability_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install finite probability and its exact checkers without a portfolio."""

    with open_exact_domain_services(
        tmp_path / "state",
        build_finite_probability_bundle(),
    ) as services:
        yield services
