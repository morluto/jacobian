"""Native execution failures retain the CLI's nonzero, no-result path."""

from typing import Never

import pytest
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
