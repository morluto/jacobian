"""Shared fail-closed parametrization for non-conclusion execution states."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from jacobian.contracts.results import ExecutionStatus

NON_CONCLUSION_STATUSES = (
    ExecutionStatus.TIMEOUT,
    ExecutionStatus.CANCELLED,
    ExecutionStatus.ERROR,
)


@dataclass(frozen=True, slots=True)
class FailClosedCase:
    """One non-conclusion state that must not promote to VERIFIED."""

    status: ExecutionStatus
    reason: str


def fail_closed_cases(
    *,
    include_incomplete: bool = True,
) -> tuple[FailClosedCase, ...]:
    cases = [
        FailClosedCase(ExecutionStatus.TIMEOUT, "timed out before a conclusion"),
        FailClosedCase(ExecutionStatus.CANCELLED, "cancelled before a conclusion"),
        FailClosedCase(ExecutionStatus.ERROR, "errored before a conclusion"),
    ]
    if include_incomplete:
        cases.append(
            FailClosedCase(
                ExecutionStatus.COMPLETED,
                "completed without a mathematical conclusion",
            )
        )
    return tuple(cases)


def pytest_fail_closed_params(
    cases: Iterable[FailClosedCase] | None = None,
) -> pytest.MarkDecorator:
    """Parametrize a test over fail-closed non-conclusion cases."""

    selected = tuple(cases) if cases is not None else fail_closed_cases()
    return pytest.mark.parametrize(
        "fail_closed_case",
        selected,
        ids=[case.status.value for case in selected],
    )
