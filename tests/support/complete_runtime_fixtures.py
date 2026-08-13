"""Complete-runtime fixtures for owning-tier conftest import.

Owning conftests import these fixture objects into their module namespace so
pytest registers them with directory-scoped visibility. This module is not a
``pytest_plugins`` entry: unit/domain collections must not load it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.runtime.model import JacobianRuntime
from tests.support.runtime_templates import (
    build_authorized_portfolio_template,
    build_complete_portfolio_template,
)
from tests.support.state import copy_template


def _open_runtime(
    state: Path,
    *,
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.NONE,
) -> Iterator[JacobianRuntime]:
    runtime = create_runtime(state, checker_authority=checker_authority)
    try:
        yield runtime
    finally:
        runtime.close()


@pytest.fixture(scope="session")
def complete_portfolio_template(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> Path:
    """Publish one immutable fully materialized portfolio snapshot."""

    return build_complete_portfolio_template(tmp_path_factory, request)


@pytest.fixture(scope="session")
def authorized_portfolio_template(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> Path:
    """Publish one immutable snapshot with bundled checker authority."""

    return build_authorized_portfolio_template(tmp_path_factory, request)


@pytest.fixture
def fresh_complete_runtime(
    tmp_path: Path,
) -> Iterator[JacobianRuntime]:
    """Materialize a complete runtime from an empty test-owned state root."""

    yield from _open_runtime(tmp_path / "state")


@pytest.fixture
def attached_complete_runtime(
    tmp_path: Path,
    complete_portfolio_template: Path,
) -> Iterator[JacobianRuntime]:
    """Attach a complete runtime to a private copy of its immutable template."""

    state = copy_template(complete_portfolio_template, tmp_path / "state")
    yield from _open_runtime(state)


@pytest.fixture(scope="module")
def attached_complete_runtime_read_only(
    tmp_path_factory: pytest.TempPathFactory,
    complete_portfolio_template: Path,
) -> Iterator[JacobianRuntime]:
    """Module-shared attach: one private template copy for non-mutating tests."""

    state = copy_template(
        complete_portfolio_template,
        tmp_path_factory.mktemp("attached-readonly") / "state",
    )
    yield from _open_runtime(state)


@pytest.fixture
def authorized_complete_runtime(
    tmp_path: Path,
    authorized_portfolio_template: Path,
) -> Iterator[JacobianRuntime]:
    """Attach a runtime to a private, already-authorized portfolio snapshot."""

    state = copy_template(authorized_portfolio_template, tmp_path / "state")
    yield from _open_runtime(
        state,
        checker_authority=CheckerAuthorityMode.HYDRATE_EXISTING,
    )


@pytest.fixture(scope="module")
def authorized_complete_runtime_read_only(
    tmp_path_factory: pytest.TempPathFactory,
    authorized_portfolio_template: Path,
) -> Iterator[JacobianRuntime]:
    """Module-shared authorized attach for non-mutating inspection tests."""

    state = copy_template(
        authorized_portfolio_template,
        tmp_path_factory.mktemp("authorized-readonly") / "state",
    )
    yield from _open_runtime(
        state,
        checker_authority=CheckerAuthorityMode.HYDRATE_EXISTING,
    )
