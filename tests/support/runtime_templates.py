"""Session-scoped immutable complete-runtime templates.

Only storage and complete-runtime boundary tests should register this plugin.
Runtime instances are defined separately so fixture ownership stays explicit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from filelock import FileLock

from jacobian.runtime import CheckerAuthorityMode, create_runtime
from tests.support.state import (
    publish_template,
    quiesce_sqlite_template,
    worker_template_target,
)


def _template_target(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
    name: str,
) -> tuple[Path, FileLock | None]:
    shared = worker_template_target(tmp_path_factory, request, name)
    if shared is not None:
        return shared
    base = tmp_path_factory.getbasetemp()
    target = Path(base).parent / f"{name}-{Path(base).name}"
    return target, None


@pytest.fixture(scope="session")
def complete_portfolio_template(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> Path:
    """Publish one immutable fully materialized portfolio snapshot."""

    target, lock = _template_target(
        tmp_path_factory,
        request,
        "complete-portfolio-template",
    )

    def build(staging: Path) -> None:
        runtime = create_runtime(staging)
        runtime.close()
        quiesce_sqlite_template(staging)

    return publish_template(target, build, lock=lock)


@pytest.fixture(scope="session")
def authorized_portfolio_template(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> Path:
    """Publish one immutable snapshot with bundled checker authority."""

    target, lock = _template_target(
        tmp_path_factory,
        request,
        "authorized-portfolio-template",
    )

    def build(staging: Path) -> None:
        runtime = create_runtime(
            staging,
            checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
        )
        runtime.close()
        quiesce_sqlite_template(staging)

    return publish_template(target, build, lock=lock)
