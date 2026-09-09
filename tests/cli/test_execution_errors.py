"""Native execution failures retain the CLI's nonzero, no-result path."""

import json
from typing import Never

import pytest
from tests.support.rationals import rational_payload as q
from typer.testing import CliRunner

from jacobian._execution import (
    BackendFailureReason,
    ExecutionResource,
    OperationBackendError,
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
)
from jacobian.cli import app
from jacobian.math.logic import _sat


@pytest.mark.parametrize(
    "error",
    [
        OperationBackendError(BackendFailureReason.STARTUP),
        OperationResourceExhaustedError(ExecutionResource.WORK),
        OperationExecutionTimeoutError("operation deadline expired"),
    ],
)
def test_cli_execution_failure_has_no_mathematical_output(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail(*args: object, **kwargs: object) -> Never:
        raise error from RuntimeError("private marker")

    monkeypatch.setattr(_sat, "_run_sat_worker", fail)
    result = CliRunner().invoke(
        app,
        ["run", "sat.solve", "--json", '{"cnf":{"variables":["x"],"clauses":[[1]]}}'],
    )
    assert result.exit_code != 0
    assert not result.stdout.strip()
    assert "private marker" not in result.output
    assert str(error) in result.stderr


def test_cli_lp_work_exhaustion_has_no_mathematical_output() -> None:
    n, m = 18, 6
    payload = {
        "program": {
            "variables": [f"x{i}" for i in range(n)],
            "objective": [q(1)] * n,
            "coefficients": [[q((j + 1) ** i) for j in range(n)] for i in range(m)],
            "rhs": [q(-1)] * m,
        }
    }
    result = CliRunner().invoke(
        app,
        [
            "run",
            "optimization.linear.rational_optimum.compute",
            "--json",
            json.dumps(payload),
        ],
    )
    assert result.exit_code != 0
    assert not result.stdout.strip()
    assert "operation exhausted its work allowance" in result.stderr
