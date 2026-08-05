"""Bounded local execution for installed plugin capabilities."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    loads_strict_json,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.implementation import package_import_path, package_source_digest
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


class _PluginWorkerFailureCode(StrEnum):
    """Bounded operational failures emitted by the plugin worker."""

    INVALID_REQUEST = "INVALID_REQUEST"
    SOURCE_CHANGED = "SOURCE_CHANGED"
    RESPONSE_INVALID = "RESPONSE_INVALID"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"


class PluginInputError(ValueError):
    """A plugin request cannot be parsed into its domain contract."""

    def __init__(
        self,
        *,
        path: str,
        expected: str,
        actual_type: str,
    ) -> None:
        self.path = path
        self.expected = expected
        self.actual_type = actual_type
        super().__init__(
            f"invalid plugin input at {path or '/'}: expected {expected} "
            f"(received {actual_type})"
        )


class PluginResponseError(ValueError):
    """A plugin worker response is not a safe typed response envelope."""

    def __init__(self, *, failure_code: _PluginWorkerFailureCode, message: str) -> None:
        self.failure_code = failure_code
        self.safe_message = message
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PluginExecutionResult:
    """Operational result from one local plugin worker invocation."""

    status: ExecutionStatus
    output: dict[str, Any] | None
    diagnostics: str
    detail: str | None
    runtime_ms: int
    failure_code: _PluginWorkerFailureCode | None = None
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
                failure_code=_PluginWorkerFailureCode.EXECUTION_FAILED,
            )

        if completed.stdout_exceeded:
            return PluginExecutionResult(
                status=ExecutionStatus.ERROR,
                output=None,
                diagnostics=diagnostics,
                detail=_PLUGIN_OUTPUT_TOO_LARGE,
                runtime_ms=_elapsed_ms(started),
                failure_code=_PluginWorkerFailureCode.RESPONSE_INVALID,
            )
        if completed.stderr_exceeded:
            return PluginExecutionResult(
                status=ExecutionStatus.ERROR,
                output=None,
                diagnostics=diagnostics,
                detail=_PLUGIN_DIAGNOSTICS_TOO_LARGE,
                runtime_ms=_elapsed_ms(started),
                failure_code=_PluginWorkerFailureCode.RESPONSE_INVALID,
            )
        if completed.termination is ProcessTermination.START_FAILED:
            return PluginExecutionResult(
                status=ExecutionStatus.ERROR,
                output=None,
                diagnostics=diagnostics,
                detail=_PLUGIN_STOPPED,
                runtime_ms=_elapsed_ms(started),
                failure_code=_PluginWorkerFailureCode.EXECUTION_FAILED,
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
                failure_code=_PluginWorkerFailureCode.RESPONSE_INVALID,
            )
        if completed.returncode != 0:
            _LOGGER.warning(
                "plugin worker stopped: response=%r diagnostics=%s",
                output,
                diagnostics,
            )
            code = _failure_code(output)
            return PluginExecutionResult(
                status=ExecutionStatus.ERROR,
                output=None,
                diagnostics=diagnostics,
                detail=_plugin_failure_detail(output),
                runtime_ms=_elapsed_ms(started),
                failure_code=code,
                path=_optional_string(output, "path"),
                expected=_optional_string(output, "expected"),
                actual_type=_optional_string(output, "actual_type"),
            )
        if not isinstance(output, dict):
            return PluginExecutionResult(
                status=ExecutionStatus.ERROR,
                output=None,
                diagnostics=diagnostics,
                detail=_PLUGIN_UNREADABLE_RESPONSE,
                runtime_ms=_elapsed_ms(started),
                failure_code=_PluginWorkerFailureCode.RESPONSE_INVALID,
            )
        if output.get("measured_implementation_digest") != expected_digest:
            _LOGGER.warning(
                "plugin worker measured an unexpected implementation: %r",
                output,
            )
            return PluginExecutionResult(
                status=ExecutionStatus.ERROR,
                output=None,
                diagnostics=diagnostics,
                detail=_PLUGIN_CHANGED,
                runtime_ms=_elapsed_ms(started),
                failure_code=_PluginWorkerFailureCode.SOURCE_CHANGED,
            )
        response = output.get("response")
        if not isinstance(response, dict):
            return PluginExecutionResult(
                status=ExecutionStatus.ERROR,
                output=None,
                diagnostics=diagnostics,
                detail=_PLUGIN_UNREADABLE_RESPONSE,
                runtime_ms=_elapsed_ms(started),
            )
        return PluginExecutionResult(
            status=ExecutionStatus.COMPLETED,
            output=response,
            diagnostics=diagnostics,
            detail=None,
            runtime_ms=_elapsed_ms(started),
        )


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _plugin_failure_detail(output: Any) -> str:
    code = _failure_code(output)
    if code is _PluginWorkerFailureCode.SOURCE_CHANGED:
        return _PLUGIN_CHANGED
    if code is _PluginWorkerFailureCode.INVALID_REQUEST:
        return "The plugin rejected its request. Check the capability input contract and retry."
    if code is _PluginWorkerFailureCode.PROVIDER_UNAVAILABLE:
        return "The plugin provider is unavailable. Install or repair the declared provider, then retry."
    return _PLUGIN_STOPPED


def _failure_code(value: Any) -> _PluginWorkerFailureCode:
    if isinstance(value, dict):
        raw_code = value.get("error_code")
        if not isinstance(raw_code, str):
            return _PluginWorkerFailureCode.EXECUTION_FAILED
        try:
            return _PluginWorkerFailureCode(raw_code)
        except ValueError:
            pass
    return _PluginWorkerFailureCode.EXECUTION_FAILED


def _optional_string(value: Any, key: str) -> str | None:
    if isinstance(value, dict) and isinstance(value.get(key), str):
        return cast(str, value[key])
    return None


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
