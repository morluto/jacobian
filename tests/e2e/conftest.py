"""End-to-end caller-visible journey fixtures."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest
from tests.support.runtime_templates import template_target
from tests.support.state import publish_template, quiesce_sqlite_template

from jacobian.operator_lifecycle import CheckerAuthorization, initialize_state


def _quiesce_initialized_state(root: Path) -> None:
    """Checkpoint after ``initialize_state`` once SQLite releases the store."""

    last_error: sqlite3.OperationalError | None = None
    for _ in range(20):
        try:
            quiesce_sqlite_template(root)
            return
        except sqlite3.OperationalError as exc:
            last_error = exc
            time.sleep(0.05)
    assert last_error is not None
    raise last_error


@pytest.fixture(scope="session")
def initialized_authorized_state_template(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> Path:
    """Publish one immutable operator-initialized bundled-checker MCP snapshot.

    MCP serving reads ``OperationCatalog``. Complete catalog-build templates do
    not persist that snapshot, so this fixture uses ``initialize_state`` once
    and every non-install journey copies the result.
    """

    target, lock = template_target(
        tmp_path_factory,
        request,
        "initialized-authorized-state-template",
    )

    def build(staging: Path) -> None:
        initialize_state(
            staging,
            checker_authorization=CheckerAuthorization.BUNDLED,
        )
        _quiesce_initialized_state(staging)

    return publish_template(target, build, lock=lock)


__all__ = ("initialized_authorized_state_template",)
