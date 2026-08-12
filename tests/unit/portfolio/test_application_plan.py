"""Unit tests for application installation plans and receipts."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services

from jacobian.domains.matrix_lattice import build_matrix_bundle
from jacobian.portfolio.application import open_application
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


def test_scoped_matrix_receipt_is_projected_by_complete_receipt(
    tmp_path: Path,
) -> None:
    scoped_plan = ApplicationInstallPlan.scoped(
        ("matrix",),
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    )
    complete_plan = ApplicationInstallPlan.complete(
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    )

    with (
        open_application(tmp_path / "scoped", scoped_plan) as scoped,
        open_application(tmp_path / "complete", complete_plan) as complete,
    ):
        scoped_projection = scoped.receipt.domain_projection()
        complete_projection = complete.receipt.domain_projection()

        assert (
            scoped_projection["checker_authority"]
            == complete_projection["checker_authority"]
        )
        for field in (
            "domain_ids",
            "capability_ids",
            "checker_ids",
            "schema_uris",
        ):
            assert set(scoped_projection[field]) <= set(complete_projection[field])


def test_scoped_application_rejects_unknown_domain_before_opening_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    plan = ApplicationInstallPlan.scoped(("unknown-domain",))

    with pytest.raises(ValueError, match="unknown scoped domain_id"):
        open_application(root, plan)

    assert not root.exists()


def test_scoped_application_can_omit_exact_verification(tmp_path: Path) -> None:
    plan = ApplicationInstallPlan.scoped(
        ("matrix",),
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
        include_exact_verification=False,
    )

    with open_application(tmp_path / "state", plan) as application:
        catalog_ids = {
            item.capability_id
            for item in application.core.capabilities.catalog().capabilities
        }

        assert "matrix.determinant.compute" in catalog_ids
        assert not any(item.endswith(".verify") for item in catalog_ids)
        assert application.receipt.checker_ids == ()
