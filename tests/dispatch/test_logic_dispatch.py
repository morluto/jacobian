"""Dispatch boundaries for bounded logic operations."""

from __future__ import annotations

import pytest

from jacobian._execution import (
    OperationBackendError,
    OperationResourceExhaustedError,
)
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.logic import _sat, _smt, _unsat_core
from jacobian.process import BoundedProcessResult


def _positive_integer_query() -> dict[str, str]:
    return {
        "logic": "QF_LIA",
        "smtlib": (
            "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)"
        ),
    }


def _contradictory_bounds_core() -> dict[str, str]:
    return {
        "logic": "QF_LIA",
        "smtlib": (
            "(set-logic QF_LIA)\n"
            "(declare-const x Int)\n"
            "(assert (>= x 1))\n"
            "(assert (<= x 0))\n"
            "(check-sat)"
        ),
    }


def _positive_cnf_query() -> dict[str, object]:
    return {"cnf": {"variables": ["x"], "clauses": [[1]]}}


def _smt_worker_memory_exhaustion() -> BoundedProcessResult:
    return BoundedProcessResult(
        returncode=0,
        stdout=(
            b'{"kind":"execution_error","stage":"operation_execution",'
            b'"resource":"memory"}'
        ),
        stderr=b"",
        stdout_exceeded=False,
        stderr_exceeded=False,
        timed_out=False,
    )


def _sat_worker_memory_exhaustion() -> BoundedProcessResult:
    return BoundedProcessResult(
        returncode=0,
        stdout=(
            b'{"kind":"execution_error","stage":"operation_execution",'
            b'"resource":"memory"}'
        ),
        stderr=b"",
        stdout_exceeded=False,
        stderr_exceeded=False,
        timed_out=False,
    )


def _unsat_core_worker_unavailable() -> BoundedProcessResult:
    return BoundedProcessResult(
        returncode=2,
        stdout=b"",
        stderr=b"",
        stdout_exceeded=False,
        stderr_exceeded=False,
        timed_out=False,
    )


@pytest.mark.parametrize(
    ("operation_id", "payload"),
    [
        ("sat.solve", _positive_cnf_query()),
        ("smt.solve", _positive_integer_query()),
        ("smt.unsat_core", _contradictory_bounds_core()),
    ],
)
def test_dispatch_types_parser_resource_failure_as_execution_failure(
    monkeypatch: pytest.MonkeyPatch, operation_id: str, payload: dict[str, object]
) -> None:
    """A fresh request validation cannot claim parser exhaustion is malformed.

    ``math.run`` revalidates every payload. Backend parsing happens only in
    the bounded owner worker, so a resource failure remains admissible and
    surfaces as typed execution failure instead of
    ``OperationRequestValidationError``.
    """

    if operation_id == "sat.solve":
        monkeypatch.setattr(
            _sat,
            "run_bounded_process",
            lambda *_args, **_kwargs: _sat_worker_memory_exhaustion(),
        )
    elif operation_id == "smt.solve":
        monkeypatch.setattr(
            _smt,
            "run_bounded_process",
            lambda *_args, **_kwargs: _smt_worker_memory_exhaustion(),
        )
    else:
        monkeypatch.setattr(
            _unsat_core,
            "run_bounded_process",
            lambda *_args, **_kwargs: _unsat_core_worker_unavailable(),
        )

    try:
        with pytest.raises(
            OperationBackendError
            if operation_id == "smt.unsat_core"
            else OperationResourceExhaustedError
        ):
            invoke_operation(operation_id, payload, Catalog.open())
    except OperationRequestValidationError as exc:
        raise AssertionError(
            f"{operation_id} rejected a parser resource failure as caller error"
        ) from exc


def test_dispatch_reports_memory_exhaustion_for_smt_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The smt.solve execution path names the exhausted budget it translated."""

    monkeypatch.setattr(
        _smt,
        "run_bounded_process",
        lambda *_args, **_kwargs: _smt_worker_memory_exhaustion(),
    )

    with pytest.raises(OperationResourceExhaustedError) as caught:
        invoke_operation("smt.solve", _positive_integer_query(), Catalog.open())
    assert caught.value.resource == "memory"


def test_dispatch_reports_memory_exhaustion_for_sat_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _sat,
        "run_bounded_process",
        lambda *_args, **_kwargs: _sat_worker_memory_exhaustion(),
    )

    with pytest.raises(OperationResourceExhaustedError) as caught:
        invoke_operation("sat.solve", _positive_cnf_query(), Catalog.open())
    assert caught.value.resource == "memory"


def test_dispatch_types_unsat_core_initialization_failure_as_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed core worker must not turn an accepted request into an import error."""

    monkeypatch.setattr(
        _unsat_core,
        "run_bounded_process",
        lambda *_args, **_kwargs: _unsat_core_worker_unavailable(),
    )

    with pytest.raises(OperationBackendError):
        invoke_operation("smt.unsat_core", _contradictory_bounds_core(), Catalog.open())
