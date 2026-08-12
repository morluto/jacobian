"""Tests for plan-addressed complete runtime templates."""

from __future__ import annotations

from tests.support.runtime_templates import application_template_key

from jacobian.portfolio import ApplicationInstallPlan
from jacobian.runtime import CheckerAuthorityMode


def test_application_template_key_includes_complete_plan_digest() -> None:
    plan = ApplicationInstallPlan.complete(
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    )

    key = application_template_key(plan)

    assert plan.digest() in key
    assert key == f"application-{plan.digest()}"
