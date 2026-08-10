"""Unit tests for fail-closed parametrization helpers."""

from __future__ import annotations

from tests.support.fail_closed import (
    FailClosedCase,
    fail_closed_cases,
    pytest_fail_closed_params,
)

from jacobian.contracts.results import ExecutionStatus


def test_fail_closed_cases_cover_non_conclusions() -> None:
    statuses = {case.status for case in fail_closed_cases(include_incomplete=False)}
    assert statuses == {
        ExecutionStatus.TIMEOUT,
        ExecutionStatus.CANCELLED,
        ExecutionStatus.ERROR,
    }


@pytest_fail_closed_params(fail_closed_cases(include_incomplete=True))
def test_fail_closed_params_expose_non_conclusion_cases(
    fail_closed_case: FailClosedCase,
) -> None:
    assert fail_closed_case.status in {
        ExecutionStatus.TIMEOUT,
        ExecutionStatus.CANCELLED,
        ExecutionStatus.ERROR,
        ExecutionStatus.COMPLETED,
    }
    assert fail_closed_case.reason
