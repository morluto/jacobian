"""Session-scoped immutable complete-runtime template builders.

Owning-tier conftests wrap these helpers as local ``@pytest.fixture`` definitions.
Do not register this module through ``pytest_plugins``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from filelock import FileLock

from jacobian.portfolio import ApplicationInstallPlan, open_application
from jacobian.runtime import CheckerAuthorityMode
from tests.support.state import (
    publish_template,
    quiesce_sqlite_template,
    worker_template_target,
)


def application_template_key(plan: ApplicationInstallPlan) -> str:
    """Return the immutable cache key for one application plan."""

    return f"application-{plan.digest()}"


def template_target(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
    name: str,
) -> tuple[Path, FileLock | None]:
    """Resolve the shared or worker-local template publication target."""

    shared = worker_template_target(tmp_path_factory, request, name)
    if shared is not None:
        return shared
    base = tmp_path_factory.getbasetemp()
    target = Path(base).parent / f"{name}-{Path(base).name}"
    return target, None


def build_complete_portfolio_template(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> Path:
    """Publish one immutable fully materialized portfolio snapshot."""

    plan = ApplicationInstallPlan.complete()
    target, lock = template_target(
        tmp_path_factory,
        request,
        application_template_key(plan),
    )

    def build(staging: Path) -> None:
        application = open_application(staging, plan)
        application.close()
        quiesce_sqlite_template(staging)

    return publish_template(target, build, lock=lock)


def build_authorized_portfolio_template(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> Path:
    """Publish one immutable snapshot with bundled checker authority."""

    plan = ApplicationInstallPlan.complete(
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    )
    target, lock = template_target(
        tmp_path_factory,
        request,
        application_template_key(plan),
    )

    def build(staging: Path) -> None:
        application = open_application(staging, plan)
        application.close()
        quiesce_sqlite_template(staging)

    return publish_template(target, build, lock=lock)
