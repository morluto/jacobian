"""Polynomial provider fixtures backed by the narrow production installer."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.component.providers.polynomial.polynomial_capabilities_support import (
    PolynomialTestServices,
    open_polynomial_services,
)
from tests.component.providers.polynomial.polynomial_normalization_support import (
    PolynomialNormalizationTestServices,
    open_polynomial_normalization_services,
)

from jacobian.runtime.config import CheckerAuthorityMode


@pytest.fixture
def polynomial_services(tmp_path: Path) -> Iterator[PolynomialTestServices]:
    with open_polynomial_services(tmp_path / "state") as services:
        yield services


@pytest.fixture
def authorized_polynomial_services(
    tmp_path: Path,
) -> Iterator[PolynomialTestServices]:
    with open_polynomial_services(
        tmp_path / "state",
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as services:
        yield services


@pytest.fixture
def polynomial_normalization_services(
    tmp_path: Path,
) -> Iterator[PolynomialNormalizationTestServices]:
    with open_polynomial_normalization_services(tmp_path / "state") as services:
        yield services


@pytest.fixture
def authorized_polynomial_normalization_services(
    tmp_path: Path,
) -> Iterator[PolynomialNormalizationTestServices]:
    with open_polynomial_normalization_services(
        tmp_path / "state",
        with_checker=True,
    ) as services:
        yield services
