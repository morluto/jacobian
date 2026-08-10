"""Unit tests for fail-closed parametrization helpers."""

from __future__ import annotations

from tests.support.fail_closed import fail_closed_cases

from jacobian.contracts.results import ExecutionStatus


def test_fail_closed_cases_cover_non_conclusions() -> None:
    statuses = {case.status for case in fail_closed_cases(include_incomplete=False)}
    assert statuses == {
        ExecutionStatus.TIMEOUT,
        ExecutionStatus.CANCELLED,
        ExecutionStatus.ERROR,
    }
