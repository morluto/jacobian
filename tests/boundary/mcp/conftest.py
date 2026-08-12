"""Resource-aware fixtures for MCP boundary tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.support.resource_contracts import (
    IsolationClass,
    ResourceKind,
    resource_fixture,
)


@pytest.fixture
@resource_fixture(
    resources={ResourceKind.MCP, ResourceKind.SQLITE},
    isolation=IsolationClass.LIFECYCLE_OWNER,
    setup_affinity="mcp",
)
def mcp_server_state(tmp_path: Path) -> Iterator[Path]:
    """Provide private state for callers that construct an MCP server."""

    yield tmp_path


__all__ = ["mcp_server_state"]
