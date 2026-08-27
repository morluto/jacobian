"""Dispatch boundaries for bounded logic operations."""

from __future__ import annotations

import pytest

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
            b'{"outcome":"UNKNOWN","model_smtlib":null,'
            b'"exhausted":"memory","detail":"the bounded solver memory '
            b'budget was exhausted"}'
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
            b'{"outcome":"UNKNOWN","assignment":null,'
            b'"exhausted":"memory","detail":"the bounded solver memory '
            b'budget was exhausted"}'
        ),
        stderr=b"",
        stdout_exceeded=False,
        stderr_exceeded=False,
        timed_out=False,
    )


def _unsat_core_worker_unavailable() -> BoundedProcessResult:
    return BoundedProcessResult(
        returncode=None,
        stdout=b"",
        stderr=b"",
        stdout_exceeded=False,
        stderr_exceeded=False,
        timed_out=True,
    )


@pytest.mark.parametrize(
    ("operation_id", "payload"),
    [
        ("sat.solve", _positive_cnf_query()),
        ("smt.solve", _positive_integer_query()),
        ("smt.unsat_core", _contradictory_bounds_core()),
    ],
)
def test_dispatch_types_parser_resource_failure_as_execution_unknown(
    monkeypatch: pytest.MonkeyPatch, operation_id: str, payload: dict[str, object]
) -> None:
    """A fresh request validation cannot claim parser exhaustion is malformed.

    ``math.run`` revalidates every payload. Backend parsing happens only in
    the bounded owner worker, so a resource failure remains admissible and
    surfaces as typed execution UNKNOWN instead of
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
        result = invoke_operation(operation_id, payload, Catalog.open())
    except OperationRequestValidationError as exc:
        raise AssertionError(
            f"{operation_id} rejected a parser resource failure as caller error"
        ) from exc

    assert result.operation_id == operation_id
    assert result.output["outcome"] == "UNKNOWN"


def test_dispatch_reports_memory_exhaustion_for_smt_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The smt.solve execution path names the exhausted budget it translated."""

    monkeypatch.setattr(
        _smt,
        "run_bounded_process",
        lambda *_args, **_kwargs: _smt_worker_memory_exhaustion(),
    )

    result = invoke_operation("smt.solve", _positive_integer_query(), Catalog.open())

    assert result.output["outcome"] == "UNKNOWN"
    assert result.output["exhausted"] == "memory"


def test_dispatch_reports_memory_exhaustion_for_sat_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _sat,
        "run_bounded_process",
        lambda *_args, **_kwargs: _sat_worker_memory_exhaustion(),
    )

    result = invoke_operation("sat.solve", _positive_cnf_query(), Catalog.open())

    assert result.output["outcome"] == "UNKNOWN"
    assert result.output["exhausted"] == "memory"


def test_dispatch_types_unsat_core_initialization_failure_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed core worker must not turn an accepted request into an import error."""

    monkeypatch.setattr(
        _unsat_core,
        "run_bounded_process",
        lambda *_args, **_kwargs: _unsat_core_worker_unavailable(),
    )

    result = invoke_operation(
        "smt.unsat_core", _contradictory_bounds_core(), Catalog.open()
    )

    assert result.output["outcome"] == "UNKNOWN"
    assert result.output["core_indices"] == []
    assert (
        result.output["detail"]
        == "the bounded SMT core worker did not establish an outcome"
    )
