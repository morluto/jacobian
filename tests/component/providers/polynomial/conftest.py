"""Polynomial provider fixtures backed by the narrow production installer."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.component.providers.polynomial.polynomial_normalization_support import (
    PolynomialNormalizationTestServices,
    open_polynomial_normalization_services,
)
from tests.component.providers.polynomial.polynomial_operations_support import (
    PolynomialTestServices,
    open_polynomial_services,
)
from tests.support.catalog_build_options import CheckerAuthorityMode


@pytest.fixture(scope="module")
def polynomial_services(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[PolynomialTestServices]:
    with open_polynomial_services(
        tmp_path_factory.mktemp("polynomial") / "state"
    ) as services:
        yield services


@pytest.fixture(scope="module")
def authorized_polynomial_services(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[PolynomialTestServices]:
    with open_polynomial_services(
        tmp_path_factory.mktemp("authorized-polynomial") / "state",
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as services:
        yield services


@pytest.fixture
def revocable_polynomial_services(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[PolynomialTestServices]:
    """Isolate tests that intentionally mutate checker authority."""

    with open_polynomial_services(
        tmp_path_factory.mktemp("revocable-polynomial") / "state",
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as services:
        yield services


@pytest.fixture(scope="module")
def polynomial_normalization_services(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[PolynomialNormalizationTestServices]:
    with open_polynomial_normalization_services(
        tmp_path_factory.mktemp("polynomial-normalization") / "state"
    ) as services:
        yield services


@pytest.fixture(scope="module")
def authorized_polynomial_normalization_services(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[PolynomialNormalizationTestServices]:
    with open_polynomial_normalization_services(
        tmp_path_factory.mktemp("authorized-polynomial-normalization") / "state",
        with_checker=True,
    ) as services:
        yield services
