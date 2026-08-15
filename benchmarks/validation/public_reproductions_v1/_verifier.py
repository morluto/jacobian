"""Verifier execution and failure normalization.

Runs a task verifier in a fresh child interpreter and normalizes any
execution failure into the canonical zero-reward record so generic tests
always observe a deterministic reward artifact.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.validation._verifier_child import (
    VerifierExecutionError,
    VerifierOutput,
    run_verifier_in_child,
)


def _run_verifier(task: Path, app: Path, logs: Path) -> VerifierOutput:
    try:
        return run_verifier_in_child(task=task, app=app, logs=logs)
    except (ValueError, VerifierExecutionError):
        return VerifierOutput(
            reward=0.0,
            details={
                "correctness": 0.0,
                "input_binding": 0.0,
                "protocol": 0.0,
                "witness_validity": 0.0,
            },
        )
