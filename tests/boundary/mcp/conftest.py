from __future__ import annotations

from pathlib import Path

import pytest

from jacobian.operator_lifecycle import CheckerAuthorization, initialize_state
from tests.support.runtime_templates import template_target
from tests.support.state import copy_template, publish_template


@pytest.fixture(scope="session")
def compiled_mcp_state(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> Path:
    target, lock = template_target(tmp_path_factory, request, "compiled-mcp-state")

    def build(staging: Path) -> None:
        initialize_state(staging, checker_authorization=CheckerAuthorization.BUNDLED)

    return publish_template(target, build, lock=lock)


@pytest.fixture
def mcp_state(
    tmp_path: Path,
    compiled_mcp_state: Path,
) -> Path:
    """Return a private catalog state for an MCP serving test."""

    return copy_template(compiled_mcp_state, tmp_path / "state")
