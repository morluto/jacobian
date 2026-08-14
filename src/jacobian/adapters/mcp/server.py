"""Local single-runtime MCP host."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from jacobian import __version__
from jacobian.adapters.mcp.context import (
    AppState,
    RuntimeAccess,
    _configured_root,
)
from jacobian.adapters.mcp.core import (
    JacobianMCPServer,
    register_core_projection,
)
from jacobian.adapters.mcp.deployment_identity import (
    DeploymentIdentity,
    load_deployment_identity,
)
from jacobian.adapters.mcp.guidance import SERVER_DESCRIPTION, SERVER_INSTRUCTIONS
from jacobian.adapters.mcp.lifecycle import (
    runtime_lifespan,
)
from jacobian.adapters.mcp.resources import register_resources
from jacobian.adapters.mcp.tooling import MCPBlockingWorkerRegistry
from jacobian.operation_catalog import OperationCatalog
from jacobian.operation_dispatcher import OperationDispatcher
from jacobian.operation_registry import OperationRegistry
from jacobian.operation_service import OperationPolicy
from jacobian.portfolio.assembler import assemble_portfolio
from jacobian.portfolio.builtin import BUILTIN_OPERATION_MODULES
from jacobian.portfolio.context import create_portfolio_context
from jacobian.runtime import CheckerAuthorityMode
from jacobian.runtime.bootstrap import bootstrap_services
from jacobian.runtime.config import RuntimeOptions
from jacobian.runtime.model import JacobianRuntime
from jacobian.runtime.portfolio import PortfolioResources
from jacobian.runtime.services import build_runtime_services


def create_server(
    state_dir: str | Path | None = None,
    *,
    operation_policy: OperationPolicy | None = None,
) -> MCPServer[AppState]:
    """Create a catalog-only host that lazily owns one execution runtime."""

    root = _configured_root(state_dir)
    policy = operation_policy or OperationPolicy()
    catalog = OperationCatalog(
        root / "metadata.sqlite3",
        policy,
        expected_package_version=__version__,
    )
    owner = _LazyLocalRuntime(
        root,
        catalog,
        operation_policy=policy,
    )
    state = AppState(
        acquire_runtime=owner.acquire,
        operation_catalog=catalog,
        worker_registry=MCPBlockingWorkerRegistry(),
    )
    return _build_server(
        state=state,
        close_owner=owner.close,
        deployment_identity=load_deployment_identity(),
    )


class _LazyLocalRuntime:
    def __init__(
        self,
        root: Path,
        catalog: OperationCatalog,
        *,
        operation_policy: OperationPolicy,
    ) -> None:
        self.root = root
        self.catalog = catalog
        self.operation_policy = operation_policy
        self._selected_runtime: JacobianRuntime | None = None
        self._complete_binding = False
        self._lock = threading.Lock()

    def acquire(self, operation_id: str | None = None) -> RuntimeAccess:
        with self._lock:
            record = (
                self.catalog.declaration_record(operation_id)
                if operation_id is not None
                else None
            )
            built_in_modules = dict(BUILTIN_OPERATION_MODULES)
            if (
                record is None or record.module not in built_in_modules
            ) and operation_id is not None:
                runtime = self._ensure_selected_runtime()
                if not self._complete_binding:
                    dispatcher = runtime.core.operations
                    if not isinstance(dispatcher, OperationDispatcher):
                        raise TypeError("lazy runtime lost its operation dispatcher")
                    dispatcher.prepare_complete_binding()
                    options = RuntimeOptions(
                        checker_authority=CheckerAuthorityMode.HYDRATE_EXISTING,
                        operation_policy=self.operation_policy,
                    )
                    context = create_portfolio_context(
                        runtime.core, runtime.services, options
                    )
                    runtime.portfolio_resources = assemble_portfolio(
                        context, runtime.services
                    )
                    self._complete_binding = True
                return RuntimeAccess(runtime)
            return RuntimeAccess(self._ensure_selected_runtime())

    def _ensure_selected_runtime(self) -> JacobianRuntime:
        if self._selected_runtime is None:
            core = bootstrap_services(
                self.root,
                RuntimeOptions(
                    checker_authority=CheckerAuthorityMode.HYDRATE_EXISTING,
                    operation_policy=self.operation_policy,
                ),
            )
            registry = OperationRegistry(self.catalog, core.binder)
            core.operations = OperationDispatcher(self.catalog, registry)
            services = build_runtime_services(core)
            self._selected_runtime = JacobianRuntime(
                core,
                services,
                PortfolioResources(),
                start_lean_warmup=lambda: None,
            )
        return self._selected_runtime

    def close(self) -> None:
        with self._lock:
            runtimes = (self._selected_runtime,)
            self._selected_runtime = None
        for runtime in runtimes:
            if runtime is not None:
                runtime.close()


def create_server_from_runtime(
    runtime: JacobianRuntime,
    *,
    close_owner: Callable[[], None],
    start_owner: Callable[[], None] | None = None,
) -> MCPServer[AppState]:
    """Register the local MCP projection over an explicitly owned runtime.

    The caller supplies lifecycle actions so deployment bootstrap and focused
    compositions exercise identical SDK registration without obscuring which
    layer owns runtime startup and teardown.
    """

    state = AppState(
        acquire_runtime=lambda _operation_id: RuntimeAccess(runtime),
        operation_catalog=runtime.core.operations,
        worker_registry=MCPBlockingWorkerRegistry(),
    )
    return _build_server(
        state=state,
        close_owner=close_owner,
        start_owner=start_owner,
        deployment_identity=load_deployment_identity(),
    )


def _build_server(
    *,
    state: AppState,
    close_owner: Callable[[], None],
    start_owner: Callable[[], None] | None = None,
    deployment_identity: DeploymentIdentity | None = None,
    token_verifier: Any | None = None,
    auth: Any | None = None,
) -> MCPServer[AppState]:
    """Register Jacobian's fixed MCP projection over one runtime owner."""

    @asynccontextmanager
    async def lifespan(server: MCPServer[AppState]) -> AsyncIterator[AppState]:
        async with runtime_lifespan(
            server,
            state=state,
            close_owner=close_owner,
            start_owner=start_owner,
        ) as active_state:
            yield active_state

    server: MCPServer[AppState] = JacobianMCPServer(
        name="jacobian",
        title="Jacobian Mathematical Workbench",
        description=SERVER_DESCRIPTION,
        instructions=SERVER_INSTRUCTIONS,
        version=__version__,
        lifespan=lifespan,
        token_verifier=token_verifier,
        auth=auth,
    )
    register_core_projection(server, state, deployment_identity)
    register_resources(server)
    return server


if __name__ == "__main__":
    from jacobian.adapters.mcp.cli import main

    main()


__all__ = ["create_server", "create_server_from_runtime"]
