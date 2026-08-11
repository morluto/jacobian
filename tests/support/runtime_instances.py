"""Function-scoped complete-runtime fixtures for owning test tiers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.runtime.model import JacobianRuntime
from tests.support.state import copy_template


@pytest.fixture
def fresh_complete_runtime(
    tmp_path: Path,
) -> Iterator[JacobianRuntime]:
    """Materialize a complete runtime from an empty test-owned state root."""

    runtime = create_runtime(tmp_path / "state")
    try:
        yield runtime
    finally:
        runtime.close()


@pytest.fixture
def attached_complete_runtime(
    tmp_path: Path,
    complete_portfolio_template: Path,
) -> Iterator[JacobianRuntime]:
    """Attach a complete runtime to a private copy of its immutable template."""

    state = copy_template(complete_portfolio_template, tmp_path / "state")
    runtime = create_runtime(state)
    try:
        yield runtime
    finally:
        runtime.close()


@pytest.fixture
def authorized_complete_runtime(
    tmp_path: Path,
    authorized_portfolio_template: Path,
) -> Iterator[JacobianRuntime]:
    """Attach a runtime to a private, already-authorized portfolio snapshot."""

    state = copy_template(authorized_portfolio_template, tmp_path / "state")
    runtime = create_runtime(
        state,
        checker_authority=CheckerAuthorityMode.HYDRATE_EXISTING,
    )
    try:
        yield runtime
    finally:
        runtime.close()
