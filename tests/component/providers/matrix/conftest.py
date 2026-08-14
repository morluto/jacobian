"""Focused matrix provider fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import open_domain_services

from jacobian.domains.matrix_lattice import matrix_operations
from jacobian.runtime.resources import RuntimeResources
from jacobian.verification.service import VerificationService


@dataclass(frozen=True, slots=True)
class MatrixRuntime:
    core: RuntimeResources
    verification: VerificationService


@pytest.fixture(scope="module")
def matrix_services(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[MatrixRuntime]:
    with open_domain_services(
        tmp_path_factory.mktemp("matrix") / "state",
        matrix_operations(),
    ) as services:
        yield MatrixRuntime(
            core=services.core,
            verification=services.installation.verification,
        )


@pytest.fixture(scope="module")
def matrix_checker_services(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[MatrixRuntime]:
    with open_exact_domain_services(
        tmp_path_factory.mktemp("matrix-checker") / "state",
        matrix_operations(),
    ) as services:
        yield MatrixRuntime(
            core=services.core,
            verification=services.installation.verification,
        )
