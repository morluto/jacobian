from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from jacobian.operator_lifecycle import CheckerAuthorization, initialize_state


@pytest.fixture(scope="session")
def compiled_mcp_state(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("compiled-mcp-state")
    initialize_state(root, checker_authorization=CheckerAuthorization.BUNDLED)
    return root


@pytest.fixture(autouse=True)
def current_mcp_state(
    request: pytest.FixtureRequest,
    compiled_mcp_state: Path,
) -> None:
    """MCP serving tests start from the explicit operator lifecycle boundary."""

    if "tmp_path" not in request.fixturenames:
        return
    root = request.getfixturevalue("tmp_path")
    shutil.copytree(compiled_mcp_state, root, dirs_exist_ok=True)
    shutil.copytree(compiled_mcp_state, root / "state", dirs_exist_ok=True)
