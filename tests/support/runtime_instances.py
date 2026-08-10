"""Undecorated complete-runtime instance factories.

Owning-tier conftests wrap these helpers as local ``@pytest.fixture`` definitions.
Do not register this module through ``pytest_plugins``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from jacobian.runtime.model import JacobianRuntime
from tests.support.runtime_profiles import (
    ATTACHED_COMPUTE,
    ATTACHED_COMPUTE_READ_ONLY,
    AUTHORIZED_VERIFY,
    AUTHORIZED_VERIFY_READ_ONLY,
    FRESH_LIFECYCLE,
    open_runtime_for,
)


def iter_fresh_complete_runtime(tmp_path: Path) -> Iterator[JacobianRuntime]:
    """Materialize a complete runtime from an empty test-owned state root."""

    yield from open_runtime_for(FRESH_LIFECYCLE, tmp_path=tmp_path)


def iter_attached_complete_runtime(
    tmp_path: Path,
    complete_portfolio_template: Path,
) -> Iterator[JacobianRuntime]:
    """Attach a complete runtime to a private copy of its immutable template."""

    yield from open_runtime_for(
        ATTACHED_COMPUTE,
        tmp_path=tmp_path,
        complete_portfolio_template=complete_portfolio_template,
    )


def iter_attached_complete_runtime_read_only(
    tmp_path: Path,
    complete_portfolio_template: Path,
) -> Iterator[JacobianRuntime]:
    """Attach once for module-shared non-mutating complete-runtime inspection."""

    yield from open_runtime_for(
        ATTACHED_COMPUTE_READ_ONLY,
        tmp_path=tmp_path,
        complete_portfolio_template=complete_portfolio_template,
    )


def iter_authorized_complete_runtime(
    tmp_path: Path,
    authorized_portfolio_template: Path,
) -> Iterator[JacobianRuntime]:
    """Attach a runtime to a private, already-authorized portfolio snapshot."""

    yield from open_runtime_for(
        AUTHORIZED_VERIFY,
        tmp_path=tmp_path,
        authorized_portfolio_template=authorized_portfolio_template,
    )


def iter_authorized_complete_runtime_read_only(
    tmp_path: Path,
    authorized_portfolio_template: Path,
) -> Iterator[JacobianRuntime]:
    """Attach once for module-shared non-mutating authorized inspection."""

    yield from open_runtime_for(
        AUTHORIZED_VERIFY_READ_ONLY,
        tmp_path=tmp_path,
        authorized_portfolio_template=authorized_portfolio_template,
    )
