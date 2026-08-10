"""Contract tests for the typed verified-domain installation seam."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.exact_domain import (
    VerifiedDomainTestSpec,
    open_exact_domain_services,
)

from jacobian.domains.matrix_lattice import build_matrix_bundle
from jacobian.runtime.config import CheckerAuthorityMode


def test_open_exact_domain_services_installs_bundle_and_verifiers(
    tmp_path: Path,
) -> None:
    bundle = build_matrix_bundle()
    with open_exact_domain_services(tmp_path / "state", bundle) as services:
        catalog_ids = {
            item.capability_id
            for item in services.core.capabilities.catalog().capabilities
        }
        assert "matrix.determinant.compute" in catalog_ids
        assert any(item.endswith(".verify") for item in catalog_ids)
        assert services.installation.authorizes_bundled_checkers


def test_open_exact_domain_services_respects_absent_authority(
    tmp_path: Path,
) -> None:
    with open_exact_domain_services(
        tmp_path / "state",
        VerifiedDomainTestSpec(bundle=build_matrix_bundle()),
        checker_authority=CheckerAuthorityMode.NONE,
    ) as services:
        catalog_ids = {
            item.capability_id
            for item in services.core.capabilities.catalog().capabilities
        }
        assert "matrix.determinant.compute" in catalog_ids
        assert not any(item.endswith(".verify") for item in catalog_ids)
        assert not services.installation.authorizes_bundled_checkers


def test_open_exact_domain_services_rejects_empty_selection(
    tmp_path: Path,
) -> None:
    with (
        pytest.raises(ValueError, match="at least one verified domain"),
        open_exact_domain_services(tmp_path / "state"),
    ):
        pass
