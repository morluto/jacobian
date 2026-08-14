"""Lean-only authorized snapshots for startup behavior tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.catalog_build_options import CheckerAuthorityMode
from tests.support.lean_runtime import (
    LeanRuntime,
    create_lean_runtime,
    publish_lean_authorized_template,
)
from tests.support.runtime_templates import template_target
from tests.support.state import copy_template


@pytest.fixture(scope="session")
def lean_authorized_template(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> Path:
    """Publish one immutable Lean-family snapshot with bundled checker authority."""

    target, lock = template_target(
        tmp_path_factory,
        request,
        "lean-authorized-template",
    )
    return publish_lean_authorized_template(target, lock=lock)


@pytest.fixture
def lean_runtime(
    tmp_path: Path,
    lean_authorized_template: Path,
) -> Iterator[LeanRuntime]:
    """Open a private hydrate of the Lean-only authorized snapshot."""

    state = copy_template(lean_authorized_template, tmp_path / "state")
    with create_lean_runtime(
        state,
        checker_authority=CheckerAuthorityMode.HYDRATE_EXISTING,
    ) as runtime:
        yield runtime


__all__ = ("lean_authorized_template", "lean_runtime")
