"""MCP resource and prompt registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ResourceNotFoundError
from pydantic import Field

from jacobian.adapters.mcp.context import AppState, _runtime
from jacobian.adapters.mcp.guidance import (
    discovery_prompt,
    evidence_check_prompt,
)
from jacobian.adapters.mcp.tooling import _run_blocking
from jacobian.contracts.artifacts import ArtifactManifest
from jacobian.contracts.common import ArtifactUri, ExperimentUri
from jacobian.contracts.discovery import (
    EnumerationAccounting,
    EnumerationStopReason,
    ExperimentSnapshot,
    ExperimentState,
)
from jacobian.contracts.results import ContractModel, Coverage, Verification
from jacobian.contracts.search import (
    SearchAccounting,
    SearchExperimentSnapshot,
    SearchStopReason,
)
from jacobian.experiments import ExperimentNotFoundError
from jacobian.storage.errors import ArtifactNotFoundError


class _ArtifactResource(ContractModel):
    artifact_uri: ArtifactUri
    manifest: ArtifactManifest
    payload: Any


class _ExperimentAccountingResource(ContractModel):
    experiment_uri: ExperimentUri
    state: ExperimentState
    stop_reason: EnumerationStopReason | SearchStopReason | None
    coverage: Coverage | None
    verification: Verification
    accounting: EnumerationAccounting | SearchAccounting


class _ExperimentScopeUnavailableResource(ContractModel):
    experiment_uri: ExperimentUri
    scope_uri: None = None


class _ExperimentScopeResource(ContractModel):
    experiment_uri: ExperimentUri
    scope_uri: ArtifactUri
    manifest: ArtifactManifest
    payload: Any


class _ExperimentArchiveUnavailableResource(ContractModel):
    experiment_uri: ExperimentUri
    archive_uri: None = None
    page_uris: tuple[ArtifactUri, ...]


class _ExperimentArchiveResource(ContractModel):
    experiment_uri: ExperimentUri
    archive_uri: ArtifactUri
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
    except (ArtifactNotFoundError, ExperimentNotFoundError) as exc:
        raise ResourceNotFoundError(
            "The requested Jacobian resource does not exist."
        ) from exc


def _register_resources_and_prompts(
    server: MCPServer[AppState],
) -> None:
    """Register all MCP resource and prompt handlers on the server."""

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

    @server.resource(
        "experiment://{experiment_id}",
        name="experiment",
        description="Read the latest durable experiment snapshot.",
        mime_type="application/json",
    )
    async def experiment_resource(
        experiment_id: str,
        ctx: Context,
    ) -> ExperimentSnapshot | SearchExperimentSnapshot:
        with _runtime(ctx) as active_runtime:
            return await _read_resource_blocking(
                active_runtime.services.experiment_router.inspect,
                f"experiment://{experiment_id}",
            )

    @server.resource(
        "experiment://{experiment_id}/accounting",
        name="experiment-accounting",
        description="Read durable enumeration accounting and assurance labels.",
        mime_type="application/json",
    )
    async def experiment_accounting_resource(
        experiment_id: str,
        ctx: Context,
    ) -> _ExperimentAccountingResource:
        with _runtime(ctx) as active_runtime:
            snapshot = await _read_resource_blocking(
                active_runtime.services.experiment_router.inspect,
                f"experiment://{experiment_id}",
            )
            return _ExperimentAccountingResource(
                experiment_uri=snapshot.experiment_uri,
                state=snapshot.state,
                stop_reason=snapshot.stop_reason,
                coverage=(
                    snapshot.coverage
                    if isinstance(snapshot, ExperimentSnapshot)
                    else None
                ),
                verification=snapshot.verification,
                accounting=snapshot.accounting,
            )

    @server.resource(
        "experiment://{experiment_id}/scope",
        name="experiment-scope",
        description="Read the current enumeration scope artifact, when available.",
        mime_type="application/json",
    )
    async def experiment_scope_resource(
        experiment_id: str,
        ctx: Context,
    ) -> _ExperimentScopeUnavailableResource | _ExperimentScopeResource:
        with _runtime(ctx) as active_runtime:
            snapshot = await _read_resource_blocking(
                active_runtime.services.experiment_router.inspect,
                f"experiment://{experiment_id}",
            )
            if (
                not isinstance(snapshot, ExperimentSnapshot)
                or snapshot.scope_uri is None
            ):
                return _ExperimentScopeUnavailableResource(
                    experiment_uri=snapshot.experiment_uri,
                )
            scope = await _read_resource_blocking(
                active_runtime.core.store.get,
                snapshot.scope_uri,
            )
            return _ExperimentScopeResource(
                experiment_uri=snapshot.experiment_uri,
                scope_uri=scope.artifact_uri,
                manifest=scope.manifest,
                payload=scope.payload,
            )

    @server.resource(
        "experiment://{experiment_id}/archive",
        name="experiment-archive",
        description="Read the immutable archive manifest and page handles.",
        mime_type="application/json",
    )
    async def experiment_archive_resource(
        experiment_id: str,
        ctx: Context,
    ) -> _ExperimentArchiveUnavailableResource | _ExperimentArchiveResource:
        with _runtime(ctx) as active_runtime:
            snapshot = await _read_resource_blocking(
                active_runtime.services.experiment_router.inspect,
                f"experiment://{experiment_id}",
            )
            if snapshot.archive_uri is None:
                return _ExperimentArchiveUnavailableResource(
                    experiment_uri=snapshot.experiment_uri,
                    page_uris=snapshot.archive_page_uris,
                )
            archive = await _read_resource_blocking(
                active_runtime.core.store.get,
                snapshot.archive_uri,
            )
            return _ExperimentArchiveResource(
                experiment_uri=snapshot.experiment_uri,
                archive_uri=archive.artifact_uri,
                manifest=archive.manifest,
                payload=archive.payload,
            )

    @server.prompt(
        name="jacobian-discover",
        title="Discover Jacobian capabilities",
        description=(
            "Guide capability discovery without choosing the agent's mathematical "
            "research strategy."
        ),
    )
    def jacobian_discover_prompt(
        task: Annotated[
            str,
            Field(description="The mathematical task or desired outcome."),
        ],
    ) -> str:
        return discovery_prompt(task)

    @server.prompt(
        name="jacobian-check-evidence",
        title="Check mathematical evidence with Jacobian",
        description=(
            "Guide selection and use of an installed independent checker without "
            "promoting unverified evidence."
        ),
    )
    def jacobian_check_evidence_prompt(
        claim: Annotated[
            str,
            Field(description="The exact mathematical claim to check."),
        ],
        artifact_uri: Annotated[
            str | None,
            Field(description="Optional artifact:// URI carrying candidate evidence."),
        ] = None,
    ) -> str:
        return evidence_check_prompt(claim, artifact_uri)
