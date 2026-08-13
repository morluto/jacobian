"""Invocation telemetry for operation dispatch."""

from __future__ import annotations

import logging
import time

from jacobian.contracts.operations import OperationResult

_LOGGER = logging.getLogger(__name__)


def log_invocation(result: OperationResult, started: float) -> None:
    elapsed_ms = round((time.monotonic() - started) * 1000)
    diagnostic_codes = (
        ",".join(diagnostic.code for diagnostic in result.diagnostics) or "-"
    )
    verification_record_uri_present = result.verification_record_uri is not None
    _LOGGER.info(
        (
            "operation invocation operation_id=%s version=%s "
            "status=%s verification_record_uri_present=%s elapsed_ms=%d diagnostics=%s"
        ),
        result.operation_id,
        result.operation_version,
        result.execution.status.value,
        verification_record_uri_present,
        elapsed_ms,
        diagnostic_codes,
        extra={
            "jacobian_operation_id": result.operation_id,
            "jacobian_operation_version": result.operation_version,
            "jacobian_execution_status": result.execution.status.value,
            "jacobian_verification_record_uri_present": verification_record_uri_present,
            "jacobian_elapsed_ms": elapsed_ms,
            "jacobian_diagnostic_codes": tuple(
                diagnostic.code for diagnostic in result.diagnostics
            ),
        },
    )


__all__ = ["log_invocation"]
