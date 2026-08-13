"""MCP artifact-resource registration."""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ResourceNotFoundError

from jacobian.adapters.mcp.context import AppState, _runtime
from jacobian.contracts.artifacts import ArtifactManifest
from jacobian.contracts.common import ArtifactUri
from jacobian.contracts.results import ContractModel
from jacobian.storage.errors import ArtifactNotFoundError


class _ArtifactResource(ContractModel):
    artifact_uri: ArtifactUri
    manifest: ArtifactManifest
    payload: Any


def register_resources(
    server: MCPServer[AppState],
) -> None:
    """Register the durable artifact resource handler."""

    @server.resource(
        "artifact://sha256/{digest}",
        name="artifact",
        description="Read an immutable artifact manifest and payload.",
        mime_type="application/json",
    )
    def artifact_resource(
        digest: str,
        ctx: Context,
    ) -> _ArtifactResource:
        try:
            with _runtime(ctx) as active_runtime:
                artifact = active_runtime.core.store.get(f"artifact://sha256/{digest}")
        except ArtifactNotFoundError as exc:
            raise ResourceNotFoundError(
                "The requested Jacobian resource does not exist."
            ) from exc
        return _ArtifactResource(
            artifact_uri=artifact.artifact_uri,
            manifest=artifact.manifest,
            payload=artifact.payload,
        )


__all__ = ["register_resources"]
