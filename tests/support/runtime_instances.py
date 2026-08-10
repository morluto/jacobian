"""Function-scoped complete-runtime fixtures for owning test tiers."""

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
from tests.support.runtime_profiles import (
    ATTACHED_COMPUTE,
    AUTHORIZED_VERIFY,
    FRESH_LIFECYCLE,
    open_runtime_for,
)


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

    yield from open_runtime_for(FRESH_LIFECYCLE, tmp_path=tmp_path)


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

    yield from open_runtime_for(
        ATTACHED_COMPUTE,
        tmp_path=tmp_path,
        complete_portfolio_template=complete_portfolio_template,
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

    yield from open_runtime_for(
        AUTHORIZED_VERIFY,
        tmp_path=tmp_path,
        authorized_portfolio_template=authorized_portfolio_template,
    )
