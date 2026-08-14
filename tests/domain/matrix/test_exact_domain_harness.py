"""Contract tests for the typed verified-domain installation seam."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.catalog_build_options import CheckerAuthorityMode
from tests.support.exact_domain import open_exact_domain_services

from jacobian.domains.matrix_lattice import matrix_operations


def test_open_exact_domain_services_installs_bundle_and_verifiers(
    tmp_path: Path,
) -> None:
    bundle = matrix_operations()
    with open_exact_domain_services(tmp_path / "state", bundle) as services:
        catalog_ids = {
            item.operation_id for item in services.core.operations.snapshot().operations
        }
        assert "matrix.determinant.compute" in catalog_ids
        assert any(item.endswith(".verify") for item in catalog_ids)
        assert services.installation.authorize_bundled_checkers


def test_open_exact_domain_services_respects_absent_authority(
    tmp_path: Path,
) -> None:
    with open_exact_domain_services(
        tmp_path / "state",
        matrix_operations(),
        checker_authority=CheckerAuthorityMode.NONE,
    ) as services:
        catalog_ids = {
            item.operation_id for item in services.core.operations.snapshot().operations
        }
        assert "matrix.determinant.compute" in catalog_ids
        assert not any(item.endswith(".verify") for item in catalog_ids)
        assert not services.installation.authorize_bundled_checkers


def test_open_exact_domain_services_rejects_empty_selection(
    tmp_path: Path,
) -> None:
    with (
        pytest.raises(ValueError, match="at least one verified operation group"),
        open_exact_domain_services(tmp_path / "state"),
    ):
        pass
