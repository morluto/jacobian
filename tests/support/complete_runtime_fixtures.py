"""Decorated complete-runtime fixtures for owning-tier conftest import.

Owning conftests import these fixture objects into their module namespace so
pytest registers them with directory-scoped visibility. This module is not a
``pytest_plugins`` entry: unit/domain collections must not load it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from jacobian.runtime.model import JacobianRuntime
from tests.support.resource_contracts import (
    IsolationClass,
    ResourceKind,
    resource_fixture,
)
from tests.support.runtime_instances import (
    iter_attached_complete_runtime,
    iter_attached_complete_runtime_read_only,
    iter_authorized_complete_runtime,
    iter_authorized_complete_runtime_read_only,
    iter_fresh_complete_runtime,
)
from tests.support.runtime_templates import (
    build_authorized_portfolio_template,
    build_complete_portfolio_template,
)


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
@resource_fixture(
    resources={ResourceKind.COMPLETE_RUNTIME},
    isolation=IsolationClass.LIFECYCLE_OWNER,
    profile_key="fresh-complete-v1",
)
def fresh_complete_runtime(
    tmp_path: Path,
) -> Iterator[JacobianRuntime]:
    """Materialize a complete runtime from an empty test-owned state root."""

    yield from iter_fresh_complete_runtime(tmp_path)


@pytest.fixture
@resource_fixture(
    resources={ResourceKind.COMPLETE_RUNTIME},
    isolation=IsolationClass.PRIVATE_MUTABLE,
    profile_key="attached-complete-v1",
)
def attached_complete_runtime(
    tmp_path: Path,
    complete_portfolio_template: Path,
) -> Iterator[JacobianRuntime]:
    """Attach a complete runtime to a private copy of its immutable template."""

    yield from iter_attached_complete_runtime(tmp_path, complete_portfolio_template)


@pytest.fixture(scope="module")
@resource_fixture(
    resources={ResourceKind.COMPLETE_RUNTIME},
    isolation=IsolationClass.READ_ONLY,
    share_scope="module",
    profile_key="attached-complete-readonly-v1",
)
def attached_complete_runtime_read_only(
    tmp_path_factory: pytest.TempPathFactory,
    complete_portfolio_template: Path,
) -> Iterator[JacobianRuntime]:
    """Module-shared attach: one private template copy for non-mutating tests."""

    yield from iter_attached_complete_runtime_read_only(
        tmp_path_factory.mktemp("attached-readonly"),
        complete_portfolio_template,
    )


@pytest.fixture
@resource_fixture(
    resources={
        ResourceKind.COMPLETE_RUNTIME,
        ResourceKind.AUTHORIZED_CHECKERS,
    },
    isolation=IsolationClass.PRIVATE_MUTABLE,
    profile_key="authorized-complete-v1",
)
def authorized_complete_runtime(
    tmp_path: Path,
    authorized_portfolio_template: Path,
) -> Iterator[JacobianRuntime]:
    """Attach a runtime to a private, already-authorized portfolio snapshot."""

    yield from iter_authorized_complete_runtime(tmp_path, authorized_portfolio_template)


@pytest.fixture(scope="module")
@resource_fixture(
    resources={
        ResourceKind.COMPLETE_RUNTIME,
        ResourceKind.AUTHORIZED_CHECKERS,
    },
    isolation=IsolationClass.READ_ONLY,
    share_scope="module",
    profile_key="authorized-complete-readonly-v1",
)
def authorized_complete_runtime_read_only(
    tmp_path_factory: pytest.TempPathFactory,
    authorized_portfolio_template: Path,
) -> Iterator[JacobianRuntime]:
    """Module-shared authorized attach for non-mutating inspection tests."""

    yield from iter_authorized_complete_runtime_read_only(
        tmp_path_factory.mktemp("authorized-readonly"),
        authorized_portfolio_template,
    )


COMPLETE_RUNTIME_FIXTURE_NAMES = (
    "complete_portfolio_template",
    "authorized_portfolio_template",
    "fresh_complete_runtime",
    "attached_complete_runtime",
    "attached_complete_runtime_read_only",
    "authorized_complete_runtime",
    "authorized_complete_runtime_read_only",
)
