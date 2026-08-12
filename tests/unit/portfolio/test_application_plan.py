"""Unit tests for application installation plans and receipts."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services

from jacobian.domains.matrix_lattice import build_matrix_bundle
from jacobian.portfolio.application_plan import (
    ApplicationInstallPlan,
    receipt_from_installed_bundles,
)
from jacobian.runtime.config import CheckerAuthorityMode


def test_complete_plan_rejects_domain_ids() -> None:
    with pytest.raises(ValueError, match="complete plans"):
        ApplicationInstallPlan(
            kind="complete",
            domain_ids=("matrix",),
            checker_authority=CheckerAuthorityMode.NONE,
        )


def test_scoped_plan_requires_domain_ids() -> None:
    with pytest.raises(ValueError, match="scoped plans"):
        ApplicationInstallPlan.scoped(())


def test_plan_digest_is_stable() -> None:
    left = ApplicationInstallPlan.scoped(
        ("matrix", "topology"),
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    )
    right = ApplicationInstallPlan.scoped(
        ("matrix", "topology"),
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    )
    assert left.digest() == right.digest()
    assert left.digest() != ApplicationInstallPlan.complete().digest()


def test_exact_domain_services_emit_installation_receipt(tmp_path: Path) -> None:
    with open_exact_domain_services(
        tmp_path / "state",
        build_matrix_bundle(),
    ) as services:
        assert services.plan.kind == "scoped"
        assert services.plan.domain_ids == ("matrix",)
        assert services.receipt.plan_digest == services.plan.digest()
        assert "matrix" in services.receipt.domain_ids
        assert services.receipt.capability_ids
        assert services.receipt.checker_ids
        assert services.receipt.checker_authority == "INSTALL_BUNDLED"
        projection = services.receipt.domain_projection()
        assert projection["domain_ids"] == ["matrix"]
        rebuilt = receipt_from_installed_bundles(
            services.plan,
            services.bundles,
            checker_ids=services.receipt.checker_ids,
        )
        assert rebuilt.domain_projection() == projection
