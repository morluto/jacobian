"""Finite-poset domain fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.domains.posets import finite_poset_operations


@pytest.fixture(scope="module")
def poset_services(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path_factory.mktemp("poset") / "state",
        finite_poset_operations(),
    ) as services:
        yield services


@pytest.fixture(scope="module")
def verified_poset_services(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[DomainTestServices]:
    with open_exact_domain_services(
        tmp_path_factory.mktemp("verified-poset") / "state",
        finite_poset_operations(),
    ) as services:
        yield services
