"""Focused production MCP compositions owned by the MCP boundary suite."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from mcp.server import MCPServer

from jacobian.adapters.mcp.context import AppState
from jacobian.adapters.mcp.server import create_server_from_runtime
from jacobian.domain_bundles import DomainBundle
from jacobian.runtime import CheckerAuthorityMode
from jacobian.runtime.model import JacobianRuntime
from jacobian.runtime.portfolio import PortfolioResources
from tests.support.services import open_domain_services


@contextmanager
def open_focused_mcp_server(
    root: Path,
    *bundles: DomainBundle,
) -> Iterator[MCPServer[AppState]]:
    """Open the real MCP projection over selected production domain bundles."""

    with open_domain_services(
        root,
        *bundles,
        checker_authority=CheckerAuthorityMode.NONE,
    ) as services:
        runtime = JacobianRuntime(
            services.core,
            services.application,
            PortfolioResources(),
            lambda: None,
        )
        yield create_server_from_runtime(
            runtime,
            close_owner=lambda: None,
        )
