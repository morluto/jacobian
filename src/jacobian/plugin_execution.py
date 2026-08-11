"""Bounded local execution for installed plugin capabilities."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    loads_strict_json,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.implementation import package_import_path, package_source_digest
from jacobian.plugin_protocol import (
    PluginWorkerContractFailure,
    PluginWorkerFailure,
    PluginWorkerFailureCode,
    PluginWorkerProtocolError,
    PluginWorkerSuccess,
    parse_plugin_worker_response,
)
from jacobian.process_policy import (
    ProcessRequest,
    ProcessTermination,
    execute_process,
)
from jacobian.worker_environment import worker_environment

_LOGGER = logging.getLogger(__name__)

_PLUGIN_TIMEOUT = (
    "The plugin did not finish within the allowed time. "
    "Retry with a larger time budget or a smaller request."
)
_PLUGIN_OUTPUT_TOO_LARGE = (
    "The plugin returned too much data. Retry with a smaller request."
)
_PLUGIN_DIAGNOSTICS_TOO_LARGE = (
    "The plugin produced too many diagnostics. Retry with a smaller request "
    "and inspect the local plugin log if the limit is reached again."
)
_PLUGIN_UNREADABLE_RESPONSE = (
    "The plugin returned an unreadable response. Retry once; "
    "if it happens again, inspect the local plugin log."
)
_PLUGIN_CHANGED = (
    "The plugin changed after it was registered. "
    "Reload Jacobian to register the current plugin version, then retry."
)
_PLUGIN_STOPPED = (
    "The plugin stopped before returning a result. Retry once; "
    "if it happens again, inspect the local plugin log."
)


@dataclass(frozen=True, slots=True)
class PluginExecutionResult:
    """Operational result from one local plugin worker invocation."""

    status: ExecutionStatus
    output: dict[str, Any] | None
    diagnostics: str
    detail: str | None
    runtime_ms: int
    failure_code: PluginWorkerFailureCode | None = None
    path: str | None = None
    expected: str | None = None
    actual_type: str | None = None


class PluginExecutor:
    """Run operator-installed local code with bounded, fail-closed outcomes.

    The child-process boundary limits elapsed time, output, and descendant
    lifetime. It is not a security sandbox and does not make plugin results
    mathematically trusted.
    """

    def __init__(
        self,
        *,
        max_output_bytes: int = 4 * 1024 * 1024,
        max_diagnostic_bytes: int = 1024 * 1024,
    ) -> None:
        self.max_output_bytes = max_output_bytes
        self.max_diagnostic_bytes = max_diagnostic_bytes

    def run(
        self,
        *,
        entrypoint: str,
        implementation_digest: str | None = None,
        request: dict[str, Any],
        timeout_seconds: float,
    ) -> PluginExecutionResult:
        """Execute a capability only if the worker measures the expected source."""

        started = time.monotonic()
        expected_digest = implementation_digest or package_source_digest(entrypoint)
        environment = worker_environment(
            overrides={"PYTHONPATH": package_import_path(entrypoint)}
        )
        completed = execute_process(
            ProcessRequest(
                executable=sys.executable,
                arguments=(
                    "-m",
                    "jacobian.plugin_worker",
                    entrypoint,
                    expected_digest,
                ),
                stdin_bytes=canonicalize_json(request),
                timeout_seconds=timeout_seconds,
                environment=environment,
                cwd=str(Path.cwd()),
                stdout_limit_bytes=self.max_output_bytes,
                stderr_limit_bytes=self.max_diagnostic_bytes,
            )
        )
        diagnostics = _bounded_text(
            completed.stderr,
            limit=self.max_diagnostic_bytes,
        )
        if completed.termination is ProcessTermination.TIMED_OUT:
            return PluginExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                output=None,
                diagnostics=diagnostics,
                detail=_PLUGIN_TIMEOUT,
                runtime_ms=_elapsed_ms(started),
                failure_code=PluginWorkerFailureCode.EXECUTION_FAILED,
            )

        if completed.stdout_exceeded:
            return PluginExecutionResult(
                status=ExecutionStatus.ERROR,
                output=None,
                diagnostics=diagnostics,
                detail=_PLUGIN_OUTPUT_TOO_LARGE,
                runtime_ms=_elapsed_ms(started),
                failure_code=PluginWorkerFailureCode.RESPONSE_INVALID,
            )
        if completed.stderr_exceeded:
            return PluginExecutionResult(
                status=ExecutionStatus.ERROR,
                output=None,
                diagnostics=diagnostics,
                detail=_PLUGIN_DIAGNOSTICS_TOO_LARGE,
                runtime_ms=_elapsed_ms(started),
                failure_code=PluginWorkerFailureCode.RESPONSE_INVALID,
            )
        if completed.termination is ProcessTermination.START_FAILED:
            return PluginExecutionResult(
                status=ExecutionStatus.ERROR,
                output=None,
                diagnostics=diagnostics,
                detail=_PLUGIN_STOPPED,
                runtime_ms=_elapsed_ms(started),
                failure_code=PluginWorkerFailureCode.EXECUTION_FAILED,
            )
        try:
            output = loads_strict_json(completed.stdout)
        except CanonicalizationError:
            return PluginExecutionResult(
                status=ExecutionStatus.ERROR,
                output=None,
                diagnostics=diagnostics,
                detail=_PLUGIN_UNREADABLE_RESPONSE,
                runtime_ms=_elapsed_ms(started),
                failure_code=PluginWorkerFailureCode.RESPONSE_INVALID,
            )
        try:
            response = parse_plugin_worker_response(output)
        except PluginWorkerProtocolError:
            return PluginExecutionResult(
                status=ExecutionStatus.ERROR,
                output=None,
                diagnostics=diagnostics,
                detail=_PLUGIN_UNREADABLE_RESPONSE,
                runtime_ms=_elapsed_ms(started),
                failure_code=PluginWorkerFailureCode.RESPONSE_INVALID,
            )
        return _plugin_result_from_response(
            response=response,
            returncode=completed.returncode,
            expected_digest=expected_digest,
            diagnostics=diagnostics,
            runtime_ms=_elapsed_ms(started),
        )


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _plugin_result_from_response(
    *,
    response: PluginWorkerSuccess | PluginWorkerFailure,
    returncode: int | None,
    expected_digest: str,
    diagnostics: str,
    runtime_ms: int,
) -> PluginExecutionResult:
    if returncode != 0 and isinstance(response, PluginWorkerSuccess):
        _LOGGER.warning(
            "plugin worker exited unsuccessfully with a success envelope: %r",
            response,
        )
        return PluginExecutionResult(
            status=ExecutionStatus.ERROR,
            output=None,
            diagnostics=diagnostics,
            detail=_PLUGIN_STOPPED,
            runtime_ms=runtime_ms,
            failure_code=PluginWorkerFailureCode.EXECUTION_FAILED,
        )
    if not isinstance(response, PluginWorkerSuccess):
        if returncode != 0:
            _LOGGER.warning(
                "plugin worker stopped: response=%r diagnostics=%s",
                response,
                diagnostics,
            )
        return PluginExecutionResult(
            status=ExecutionStatus.ERROR,
            output=None,
            diagnostics=diagnostics,
            detail=_plugin_failure_detail(response),
            runtime_ms=runtime_ms,
            failure_code=PluginWorkerFailureCode(response.error_code),
            path=(
                response.path
                if isinstance(response, PluginWorkerContractFailure)
                else None
            ),
            expected=(
                response.expected
                if isinstance(response, PluginWorkerContractFailure)
                else None
            ),
            actual_type=(
                response.actual_type
                if isinstance(response, PluginWorkerContractFailure)
                else None
            ),
        )
    if response.measured_implementation_digest != expected_digest:
        _LOGGER.warning(
            "plugin worker measured an unexpected implementation: %r",
            response,
        )
        return PluginExecutionResult(
            status=ExecutionStatus.ERROR,
            output=None,
            diagnostics=diagnostics,
            detail=_PLUGIN_CHANGED,
            runtime_ms=runtime_ms,
            failure_code=PluginWorkerFailureCode.SOURCE_CHANGED,
        )
    return PluginExecutionResult(
        status=ExecutionStatus.COMPLETED,
        output=response.response,
        diagnostics=diagnostics,
        detail=None,
        runtime_ms=runtime_ms,
    )


def _plugin_failure_detail(output: PluginWorkerFailure) -> str:
    code = PluginWorkerFailureCode(output.error_code)
    if code is PluginWorkerFailureCode.SOURCE_CHANGED:
        return _PLUGIN_CHANGED
    if code is PluginWorkerFailureCode.INVALID_REQUEST:
        return "The plugin rejected its request. Check the capability input contract and retry."
    return _PLUGIN_STOPPED


def _bounded_text(value: bytes | str | None, *, limit: int) -> str:
    if value is None:
        return ""
    raw = value.encode("utf-8", errors="replace") if isinstance(value, str) else value
    if len(raw) <= limit:
        selected = raw
    else:
        marker = b"\n...[truncated]"
        selected = raw[: max(0, limit - len(marker))] + marker[:limit]
    return selected.decode("utf-8", errors="replace")
