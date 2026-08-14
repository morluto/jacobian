"""Finite-field domain fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.support.exact_domain import (
    VerifiedDomainTestServices,
    open_exact_domain_services,
)

from jacobian.domains.finite_fields import finite_field_operations


@pytest.fixture(scope="module")
def finite_field_services(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[VerifiedDomainTestServices]:
    with open_exact_domain_services(
        tmp_path_factory.mktemp("finite-field") / "state",
        finite_field_operations(),
    ) as services:
        yield services
