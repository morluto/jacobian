"""Bounded subprocess execution for authorized checker workers."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jacobian.bounded_process import ProcessResourceLimits
from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    loads_strict_json,
)
from jacobian.contracts.capabilities import CapabilityProviderRuntime
from jacobian.contracts.checkers import CheckerDecision, CheckerManifest
from jacobian.process_policy import ProcessRequest, ProcessTermination, execute_process
from jacobian.verification._helpers import (
    _CHECKER_CANCELLED,
    _CHECKER_CHANGED,
    _CHECKER_DIAGNOSTICS_TOO_LARGE,
    _CHECKER_INVALID_DECISION,
    _CHECKER_OUTPUT_TOO_LARGE,
    _CHECKER_STOPPED,
    _CHECKER_UNREADABLE_RESPONSE,
    _checker_failure_detail,
)
from jacobian.verification.checker_protocol import (
    CheckerWorkerDecisionError,
    CheckerWorkerFailure,
    CheckerWorkerProtocolError,
    parse_checker_worker_response,
)
from jacobian.verification.errors import (
    CheckerExecutionCancelledError,
    CheckerExecutionError,
)
from jacobian.worker_environment import worker_environment

_LOGGER = logging.getLogger(__name__)


@dataclass
class BoundedCheckerExecutor:
    """Execute one checker worker within its declared time and output bounds."""

    checker_timeout_seconds: float
    max_checker_output_bytes: int
    max_checker_diagnostic_bytes: int

    def execute(
        self,
        *,
        manifest: CheckerManifest,
        expected_implementation_digest: str,
        request: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> CheckerDecision:
        """Run one worker and return only its identity-bound decision."""

        environment = worker_environment(
            extra_variables=(
                "ELAN_HOME",
                "JACOBIAN_CHECKER_EXECUTABLE",
                "JACOBIAN_CHECKER_RUNTIME_DIGEST",
                "JACOBIAN_CHECKER_LAKE_DIGEST",
                "JACOBIAN_LEAN_RUNTIME",
            )
        )
        declared = manifest.sandbox
        effective_timeout = min(
            30 if timeout_seconds is None else timeout_seconds,
            self.checker_timeout_seconds,
            declared.max_wall_seconds,
        )
        arguments = (
            "-m",
            "jacobian.checker_worker",
            canonicalize_json(manifest.model_dump(mode="json")).decode("utf-8"),
        )
        completed = execute_process(
            ProcessRequest(
                executable=sys.executable,
                arguments=arguments,
                stdin_bytes=canonicalize_json(request),
                timeout_seconds=effective_timeout,
                environment=environment,
                cwd=str(Path.cwd()),
                stdout_limit_bytes=min(
                    self.max_checker_output_bytes,
                    declared.max_stdout_bytes,
                ),
                stderr_limit_bytes=min(
                    self.max_checker_diagnostic_bytes,
                    declared.max_stderr_bytes,
                ),
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=min(
                        declared.max_cpu_seconds,
                        max(1, int(effective_timeout) + 1),
                    ),
                    address_space_bytes=declared.max_address_space_bytes,
                ),
            )
        )
        if completed.termination is ProcessTermination.TIMED_OUT:
            raise TimeoutError("checker execution timed out")
        if completed.termination is ProcessTermination.CANCELLED:
            raise CheckerExecutionCancelledError(_CHECKER_CANCELLED)
        if completed.stdout_exceeded:
            raise CheckerExecutionError(_CHECKER_OUTPUT_TOO_LARGE)
        if completed.stderr_exceeded:
            raise CheckerExecutionError(_CHECKER_DIAGNOSTICS_TOO_LARGE)
        return self._validate_response(
            completed,
            expected_implementation_digest,
            manifest.provider_runtime,
        )

    @staticmethod
    def _validate_response(
        completed: Any,
        expected_digest: str,
        provider_runtime: CapabilityProviderRuntime | None,
    ) -> CheckerDecision:
        try:
            response = parse_checker_worker_response(
                loads_strict_json(completed.stdout)
            )
        except CanonicalizationError as exc:
            raise CheckerExecutionError(_CHECKER_UNREADABLE_RESPONSE) from exc
        except CheckerWorkerDecisionError as exc:
            raise CheckerExecutionError(_CHECKER_INVALID_DECISION) from exc
        except CheckerWorkerProtocolError as exc:
            raise CheckerExecutionError(_CHECKER_UNREADABLE_RESPONSE) from exc
        if completed.returncode != 0:
            _LOGGER.warning(
                "checker worker stopped: response=%r diagnostics=%r",
                response,
                completed.stderr,
            )
            detail = (
                _checker_failure_detail(response)
                if isinstance(response, CheckerWorkerFailure)
                else _CHECKER_STOPPED
            )
            raise CheckerExecutionError(detail)
        if isinstance(response, CheckerWorkerFailure):
            raise CheckerExecutionError(_checker_failure_detail(response))
        if response.measured_implementation_digest != expected_digest:
            _LOGGER.warning(
                "checker worker measured an unexpected implementation: %r",
                response,
            )
            raise CheckerExecutionError(_CHECKER_CHANGED)
        expected_runtime_digest = (
            provider_runtime.digest if provider_runtime is not None else None
        )
        if response.measured_runtime_digest != expected_runtime_digest:
            _LOGGER.warning(
                "checker worker measured an unexpected external runtime: %r",
                response,
            )
            raise CheckerExecutionError(_CHECKER_CHANGED)
        return response.decision
