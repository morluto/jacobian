"""MCP tool handlers for the operation surface."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError

from jacobian._execution import (
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationResourceExhaustedError,
    RequestCancellationSignal,
)
from jacobian.backends import BackendUnavailableError, check_backend
from jacobian.catalog.models import (
    OperationId,
    OperationResourceAdmissionError,
    OperationResult,
)
from jacobian.dispatch import (
    OperationDomainValidationError,
    OperationExecutionTimeoutError,
    OperationRequestValidationError,
    _OperationResolutionError,
    execute_operation,
)
from jacobian.mcp.models import (
    OperationCursor,
    OperationDiscoveryError,
    OperationDiscoveryErrorDetail,
    OperationFindOperationId,
    OperationFindResponse,
    OperationInspectionResult,
    OperationInvalidRequestData,
    OperationMatchLimit,
    OperationNamespace,
    OperationNeed,
    OperationResourceAdmissionData,
    OperationValidationIssue,
)
from jacobian.mcp.projections import _operation_match_response
from jacobian.mcp.runtime import (
    AppState,
    _authorize,
    _catalog,
)

_MAX_VALIDATION_ERRORS = 64
_MAX_VALIDATION_LOCATION_COMPONENTS = 32
_MAX_VALIDATION_LOCATION_LENGTH = 128
_FIND_QUERY_HASH_HEX_LENGTH = 16
_FIND_QUERY_LOG_KEY = secrets.token_bytes(32)

logger = logging.getLogger(__name__)


def _find_invalid_request_error(
    message: str, *, hint: str, locations: tuple[str, ...]
) -> ToolError:
    diagnostic = {
        "code": "INVALID_REQUEST",
        "stage": "operation_discovery",
        "errors": [
            {
                "location": list(locations),
                "code": "invalid_request",
                "message": message,
            }
        ],
        "hint": hint,
        "message": message,
    }
    return ToolError(json.dumps(diagnostic, separators=(",", ":")))


def math_find(
    query: OperationNeed | None = None,
    operation_id: OperationFindOperationId | None = None,
    namespace: OperationNamespace = None,
    limit: OperationMatchLimit | None = None,
    cursor: OperationCursor = None,
    *,
    ctx: Context[AppState, Any],
) -> OperationFindResponse:
    if query is None and operation_id is None:
        raise _find_invalid_request_error(
            "Provide exactly one of query or operation_id.",
            hint="Use query to search, or operation_id to inspect one operation.",
            locations=("query", "operation_id"),
        )
    if query is not None and operation_id is not None:
        raise _find_invalid_request_error(
            "Provide exactly one of query or operation_id, not both.",
            hint="Remove operation_id for a search, or remove query for inspection.",
            locations=("query", "operation_id"),
        )
    if operation_id is not None and (
        namespace is not None or limit is not None or cursor is not None
    ):
        raise _find_invalid_request_error(
            "namespace, limit, and cursor are only valid with query.",
            hint="Remove search options when inspecting an operation_id.",
            locations=("namespace", "limit", "cursor"),
        )

    active_catalog = _catalog(ctx)
    if query is not None:
        # The process-local HMAC key prevents log readers from recovering a
        # short caller need through an offline digest lookup.
        need_hash = hmac.new(
            _FIND_QUERY_LOG_KEY, query.encode("utf-8"), hashlib.sha256
        ).hexdigest()[:_FIND_QUERY_HASH_HEX_LENGTH]
        logger.info("math.find query_hash=%s", need_hash)
        match_response = _operation_match_response(
            active_catalog,
            need=query,
            namespace=namespace,
            limit=limit if limit is not None else 10,
            cursor=cursor,
        )
        return OperationFindResponse(root=match_response)

    assert operation_id is not None
    descriptor = active_catalog.inspect(operation_id)
    if descriptor is None:
        hint = (
            "Call math.find with a local mathematical need to match installed "
            "operations."
        )
        return OperationFindResponse(
            root=OperationDiscoveryError(
                kind="error",
                error=OperationDiscoveryErrorDetail(
                    code="UNKNOWN_OPERATION",
                    stage="operation_resolution",
                    message=f"Unknown operation: {operation_id}",
                    hint=hint,
                ),
            )
        )
    return OperationFindResponse(
        OperationInspectionResult(
            kind="operation",
            operation=descriptor,
            backend_availability=tuple(
                check_backend(name) for name in descriptor.runtime_requirements
            ),
        )
    )


def math_run(
    operation_id: OperationId,
    payload: dict[str, Any],
    *,
    ctx: Context[AppState, Any],
) -> OperationResult:
    """Run one math tool. Role comes from the tool ID."""
    _authorize(ctx)
    catalog = _catalog(ctx)
    cancellation = _request_cancellation(ctx)
    with _operation_error_boundary(operation_id):
        try:
            return execute_operation(
                operation_id,
                payload,
                catalog,
                projector=lambda selected_id, result, started: OperationResult(
                    operation_id=selected_id,
                    runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
                    output=result.model_dump(mode="json"),
                ),
                cancellation_signal=cancellation,
            )
        except _OperationResolutionError as exc:
            raise ToolError(str(exc)) from exc


@contextmanager
def _operation_error_boundary(operation_id: str) -> Iterator[None]:
    """Translate and privately log failures once for both SDK entry points."""

    try:
        yield
    except (OperationRequestValidationError, OperationDomainValidationError) as exc:
        raise _invalid_request_error(operation_id, exc) from exc
    except OperationExecutionTimeoutError as exc:
        raise _execution_tool_error(
            code="OPERATION_TIMEOUT", operation_id=operation_id, stage=exc.stage
        ) from exc
    except OperationExecutionCancelledError as exc:
        raise _execution_tool_error(
            code="OPERATION_CANCELLED", operation_id=operation_id, stage=exc.stage
        ) from exc
    except OperationResourceExhaustedError as exc:
        hint = None
        if exc.resource == "work":
            if operation_id == "smt.unsat_core":
                hint = "Adjust rlimit within its admitted range; timeout_ms does not increase the work allowance."
            elif operation_id in {"sat.solve", "smt.solve"}:
                hint = "timeout_ms does not increase the fixed work allowance."
            elif operation_id in {
                "optimization.linear.rational_optimum.compute",
                "optimization.linear.rational_general_optimum.compute",
            }:
                hint = (
                    "The exact basis search reached its fixed scalar-update allowance. "
                    "If exact primal-dual candidates are available, submit them to "
                    "optimization.linear.rational_optimality.check."
                )
            elif operation_id in {
                "graph.k_regular_subgraph.find",
                "graph.cycle.fixed_length.decide",
                "graph.subgraph_pattern.find",
                "graph.edge_colored_subgraph_pattern.find",
                "hypergraph.nonmonochromatic_vertex_coloring.q_decide",
            }:
                hint = (
                    "The bounded search prefix found no witness; this does not "
                    "establish a negative result. Reduce the search space enough "
                    "for the operation to complete within its fixed work allowance."
                )
        raise _execution_tool_error(
            code="RESOURCE_EXHAUSTED",
            operation_id=operation_id,
            stage=exc.stage,
            resource=exc.resource,
            hint=hint,
        ) from exc
    except OperationBackendError as exc:
        logger.exception(
            "operation backend failure operation_id=%s reason=%s",
            operation_id,
            exc.reason,
        )
        raise _execution_tool_error(
            code="OPERATION_FAILED", operation_id=operation_id, stage=exc.stage
        ) from exc
    except BackendUnavailableError as exc:
        raise _backend_unavailable_error(operation_id, exc) from exc
    except (MCPError, ToolError):
        raise
    except Exception as exc:
        logger.exception("unexpected operation failure operation_id=%s", operation_id)
        raise ToolError("operation execution failed") from exc


def _invalid_request_error(
    operation_id: OperationId,
    error: OperationRequestValidationError | OperationDomainValidationError,
) -> ToolError:
    """Project one owner-bound rejection without reflecting caller values."""

    issues = _bounded_validation_issues(error.errors())
    if isinstance(error, OperationResourceAdmissionError):
        data: OperationInvalidRequestData | OperationResourceAdmissionData = (
            OperationResourceAdmissionData(operation_id=operation_id, errors=issues)
        )
        message = "operation request exceeds its admitted resource envelope"
    else:
        data = OperationInvalidRequestData(operation_id=operation_id, errors=issues)
        message = "operation payload failed validation"
    diagnostic = data.model_dump(mode="json")
    diagnostic["message"] = message
    return ToolError(json.dumps(diagnostic, separators=(",", ":"), sort_keys=True))


def _backend_unavailable_error(
    operation_id: str, error: BackendUnavailableError
) -> ToolError:
    """Expose backend recovery without changing a mathematical result schema."""

    return ToolError(
        json.dumps(
            {
                "code": "BACKEND_UNAVAILABLE",
                "stage": "backend_execution",
                "operation_id": operation_id,
                "backend": error.backend,
                "required_version": error.required_version,
                "hint": error.installation,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _execution_tool_error(
    *,
    code: str,
    operation_id: str,
    stage: str,
    resource: str | None = None,
    hint: str | None = None,
) -> ToolError:
    """Project bounded operation context through the SDK's text-only tool error."""

    messages = {
        "OPERATION_TIMEOUT": "operation deadline expired",
        "OPERATION_CANCELLED": "operation was cancelled",
        "RESOURCE_EXHAUSTED": "operation exhausted its resource allowance",
        "OPERATION_FAILED": "operation backend failed",
    }
    diagnostic = {
        "code": code,
        "operation_id": operation_id,
        "stage": stage,
        "message": messages[code],
    }
    if resource is not None:
        diagnostic["resource"] = resource
    if hint is not None:
        diagnostic["hint"] = hint
    return ToolError(
        json.dumps(
            diagnostic,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _request_cancellation(ctx: Context[AppState, Any]) -> RequestCancellationSignal:
    """Return MCP 2.1's request signal through its only available SDK seam."""

    return ctx.request_context.session._request_outbound.cancel_requested


def _bounded_validation_issues(
    errors: Sequence[Mapping[str, Any]],
) -> tuple[OperationValidationIssue, ...]:
    """Build bounded field diagnostics without reflecting raw caller input."""

    issues: list[OperationValidationIssue] = []
    for error in errors[:_MAX_VALIDATION_ERRORS]:
        issues.append(
            OperationValidationIssue(
                location=_bounded_validation_location(error["loc"]),
                code=str(error["type"]),
                message=_bounded_validation_message(error["msg"]),
            )
        )
    return tuple(issues)


def _bounded_validation_location(value: Any) -> tuple[str | int, ...]:
    """Sanitize caller-controlled Pydantic locations for recovery output."""

    location: list[str | int] = []
    for component in value:
        if isinstance(component, str):
            location.append(_bounded_text(component, _MAX_VALIDATION_LOCATION_LENGTH))
        elif type(component) is int:
            location.append(component)
        if len(location) == _MAX_VALIDATION_LOCATION_COMPONENTS:
            break
    return tuple(location)


def _bounded_validation_message(value: Any) -> str:
    """Keep caller-influenced Pydantic diagnostics inside the public schema."""

    return _bounded_text(str(value), 1_024)


def _bounded_text(value: str, maximum_length: int) -> str:
    return (
        value if len(value) <= maximum_length else f"{value[: maximum_length - 3]}..."
    )
