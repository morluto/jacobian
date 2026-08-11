"""Invocation telemetry for capability dispatch."""

from __future__ import annotations

import logging
import time

from jacobian.contracts.capabilities import CapabilityResult

_LOGGER = logging.getLogger(__name__)


def log_invocation(result: CapabilityResult, started: float) -> None:
    elapsed_ms = round((time.monotonic() - started) * 1000)
    diagnostic_codes = (
        ",".join(diagnostic.code for diagnostic in result.diagnostics) or "-"
    )
    _LOGGER.info(
        (
            "capability invocation capability_id=%s version=%s "
            "status=%s assurance=%s elapsed_ms=%d diagnostics=%s"
        ),
        result.capability_id,
        result.capability_version,
        result.execution.status.value,
        result.assurance.level.value,
        elapsed_ms,
        diagnostic_codes,
        extra={
            "jacobian_capability_id": result.capability_id,
            "jacobian_capability_version": result.capability_version,
            "jacobian_execution_status": result.execution.status.value,
            "jacobian_assurance_level": result.assurance.level.value,
            "jacobian_elapsed_ms": elapsed_ms,
            "jacobian_diagnostic_codes": tuple(
                diagnostic.code for diagnostic in result.diagnostics
            ),
        },
    )


__all__ = ["log_invocation"]
