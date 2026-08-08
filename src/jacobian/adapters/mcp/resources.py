"""MCP resource and prompt registration."""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

from pydantic import Field

from jacobian.adapters.mcp.context import _experiment_scope_content, _resource_runtime
from jacobian.adapters.mcp.guidance import (
    discovery_prompt,
    evidence_check_prompt,
)
from jacobian.adapters.mcp.tooling import MCPBlockingWorkerRegistry, _run_blocking
from jacobian.runtime.model import JacobianRuntime

_LOGGER = logging.getLogger(__name__)


def _register_resources_and_prompts(
    server: Any,
    runtime: JacobianRuntime | None,
    tenant_router: Any,
    worker_registry: MCPBlockingWorkerRegistry,
) -> None:
    """Register all MCP resource and prompt handlers on the server."""

    @server.resource(  # type: ignore[untyped-decorator]
        "artifact://sha256/{digest}",
        name="artifact",
        description="Read an immutable artifact manifest and payload.",
        mime_type="application/json",
    )
    async def artifact_resource(
        digest: str,
    ) -> str:
        with _resource_runtime(
            runtime, tenant_router, worker_registry
        ) as active_runtime:
            artifact = await _run_blocking(
                active_runtime.core.store.get,
                f"artifact://sha256/{digest}",
            )
            return json.dumps(
                {
                    "artifact_uri": artifact.artifact_uri,
                    "manifest": artifact.manifest.model_dump(mode="json"),
                    "payload": artifact.payload,
                },
                ensure_ascii=False,
                sort_keys=True,
            )

    @server.resource(  # type: ignore[untyped-decorator]
        "experiment://{experiment_id}",
        name="experiment",
        description="Read the latest durable experiment snapshot.",
        mime_type="application/json",
    )
    async def experiment_resource(
        experiment_id: str,
    ) -> str:
        with _resource_runtime(
            runtime, tenant_router, worker_registry
        ) as active_runtime:
            snapshot = await _run_blocking(
                active_runtime.services.experiment_router.inspect,
                f"experiment://{experiment_id}",
            )
            return json.dumps(
                snapshot.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            )

    @server.resource(  # type: ignore[untyped-decorator]
        "experiment://{experiment_id}/accounting",
        name="experiment-accounting",
        description="Read durable enumeration accounting and assurance labels.",
        mime_type="application/json",
    )
    async def experiment_accounting_resource(
        experiment_id: str,
    ) -> str:
        with _resource_runtime(
            runtime, tenant_router, worker_registry
        ) as active_runtime:
            snapshot = await _run_blocking(
                active_runtime.services.experiment_router.inspect,
                f"experiment://{experiment_id}",
            )
            coverage = getattr(snapshot, "coverage", None)
            return json.dumps(
                {
                    "experiment_uri": snapshot.experiment_uri,
                    "state": snapshot.state.value,
                    "stop_reason": (
                        snapshot.stop_reason.value
                        if snapshot.stop_reason is not None
                        else None
                    ),
                    "coverage": coverage.value if coverage is not None else None,
                    "verification": snapshot.verification.value,
                    "accounting": snapshot.accounting.model_dump(mode="json"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )

    @server.resource(  # type: ignore[untyped-decorator]
        "experiment://{experiment_id}/scope",
        name="experiment-scope",
        description="Read the current enumeration scope artifact, when available.",
        mime_type="application/json",
    )
    async def experiment_scope_resource(
        experiment_id: str,
    ) -> str:
        with _resource_runtime(
            runtime, tenant_router, worker_registry
        ) as active_runtime:
            snapshot = await _run_blocking(
                active_runtime.services.experiment_router.inspect,
                f"experiment://{experiment_id}",
            )
            return await _run_blocking(
                _experiment_scope_content,
                active_runtime,
                snapshot,
            )

    @server.resource(  # type: ignore[untyped-decorator]
        "experiment://{experiment_id}/archive",
        name="experiment-archive",
        description="Read the immutable archive manifest and page handles.",
        mime_type="application/json",
    )
    async def experiment_archive_resource(
        experiment_id: str,
    ) -> str:
        with _resource_runtime(
            runtime, tenant_router, worker_registry
        ) as active_runtime:
            snapshot = await _run_blocking(
                active_runtime.services.experiment_router.inspect,
                f"experiment://{experiment_id}",
            )
            if snapshot.archive_uri is None:
                return json.dumps(
                    {
                        "experiment_uri": snapshot.experiment_uri,
                        "archive_uri": None,
                        "page_uris": list(snapshot.archive_page_uris),
                    },
                    sort_keys=True,
                )
            archive = await _run_blocking(
                active_runtime.core.store.get,
                snapshot.archive_uri,
            )
            return json.dumps(
                {
                    "experiment_uri": snapshot.experiment_uri,
                    "archive_uri": archive.artifact_uri,
                    "manifest": archive.manifest.model_dump(mode="json"),
                    "payload": archive.payload,
                },
                ensure_ascii=False,
                sort_keys=True,
            )

    @server.prompt(  # type: ignore[untyped-decorator]
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

    @server.prompt(  # type: ignore[untyped-decorator]
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


def _register_reasoning_resource(
    server: Any,
    runtime: JacobianRuntime | None,
    tenant_router: Any,
    worker_registry: MCPBlockingWorkerRegistry,
) -> None:
    """Register the optional operational reasoning-log reader."""

    @server.resource(  # type: ignore[untyped-decorator]
        "reasoning://run/{run_id}",
        name="reasoning-log",
        description="Read one tenant-isolated append-only external reasoning log.",
        mime_type="application/x-ndjson",
    )
    async def reasoning_log_resource(run_id: str) -> str:
        with _resource_runtime(
            runtime, tenant_router, worker_registry
        ) as active_runtime:
            return await _run_blocking(
                active_runtime.core.reasoning_log.inspect_jsonl,
                run_id,
            )
