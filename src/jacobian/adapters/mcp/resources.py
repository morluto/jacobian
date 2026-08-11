"""MCP artifact-resource registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ResourceNotFoundError

from jacobian.adapters.mcp.context import AppState, _runtime
from jacobian.adapters.mcp.tooling import _run_blocking
from jacobian.contracts.artifacts import ArtifactManifest
from jacobian.contracts.common import ArtifactUri
from jacobian.contracts.results import ContractModel
from jacobian.storage.errors import ArtifactNotFoundError


class _ArtifactResource(ContractModel):
    artifact_uri: ArtifactUri
    manifest: ArtifactManifest
    payload: Any


async def _read_resource_blocking[ResultT](
    function: Callable[..., ResultT],
    /,
    *args: Any,
) -> ResultT:
    """Run a store read and preserve the SDK's native not-found semantics."""

    try:
        return await _run_blocking(function, *args)
    except ArtifactNotFoundError as exc:
        raise ResourceNotFoundError(
            "The requested Jacobian resource does not exist."
        ) from exc


def _register_resources(
    server: MCPServer[AppState],
) -> None:
    """Register the durable artifact resource handler."""

    @server.resource(
        "artifact://sha256/{digest}",
        name="artifact",
        description="Read an immutable artifact manifest and payload.",
        mime_type="application/json",
    )
    async def artifact_resource(
        digest: str,
        ctx: Context,
    ) -> _ArtifactResource:
        with _runtime(ctx) as active_runtime:
            artifact = await _read_resource_blocking(
                active_runtime.core.store.get,
                f"artifact://sha256/{digest}",
            )
            return _ArtifactResource(
                artifact_uri=artifact.artifact_uri,
                manifest=artifact.manifest,
                payload=artifact.payload,
            )
