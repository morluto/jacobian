"""A healthy inconclusive solver answer remains a successful mathematical result."""

import json
from collections.abc import Callable, Mapping
from typing import Any

import pytest
import z3

from jacobian.math.logic import _sat, _smt, _unsat_core
from jacobian.math.logic._cnf import CanonicalCnf
from jacobian.process import BoundedProcessResult


@pytest.mark.parametrize("operation_id", ["sat.solve", "smt.solve", "smt.unsat_core"])
def test_healthy_unknown_survives_worker_decoding(
    monkeypatch: pytest.MonkeyPatch, operation_id: str
) -> None:
    monkeypatch.setattr(z3.Solver, "check", lambda *args: z3.unknown)
    monkeypatch.setattr(z3.Solver, "reason_unknown", lambda *args: "incomplete theory")
    owner: Any
    request: Any
    solve: Callable[[Any], Any]
    response: Mapping[str, object]
    if operation_id == "sat.solve":
        owner = _sat
        request = _sat.SatSolveRequest(
            cnf=CanonicalCnf(variables=("x",), clauses=((1,),))
        )
        solve = _sat.solve_sat
        response = _sat._solve_sat_kernel(
            cnf=request.cnf, timeout_ms=request.timeout_ms
        )
    elif operation_id == "smt.solve":
        owner = _smt
        request = _smt.SmtSolveRequest(
            logic=_smt.SmtLogic.QF_LIA,
            smtlib="(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
        )
        solve = _smt.solve_smt
        response = _smt._solve_smt_kernel(
            logic=request.logic.value,
            smtlib=request.smtlib,
            timeout_ms=request.timeout_ms,
        )
    else:
        owner = _unsat_core
        request = _unsat_core.SmtUnsatCoreRequest(
            logic=_smt.SmtLogic.QF_LIA,
            smtlib="(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
        )
        solve = _unsat_core.compute_smt_unsat_core
        response = _unsat_core._unsat_core_worker_kernel(request)
    monkeypatch.setattr(
        owner,
        "run_bounded_process",
        lambda *args, **kwargs: BoundedProcessResult(
            0, json.dumps(response).encode(), b"", False, False, False
        ),
    )
    result = solve(request)
    assert result.outcome == "UNKNOWN"
    assert result.detail == "the solver returned an inconclusive answer"
    assert result.model_dump().get("assignment") is None
    assert result.model_dump().get("model_smtlib") is None
    assert not result.model_dump().get("core_indices")


@pytest.mark.parametrize("sat", [False, True])
def test_solver_phase_preserves_parent_timeout(
    monkeypatch: pytest.MonkeyPatch,
    sat: bool,
) -> None:
    from jacobian._execution import OperationExecutionTimeoutError

    error = OperationExecutionTimeoutError("operation deadline expired")

    def timeout(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(_sat if sat else _smt, "_solver_settings", timeout)
    with pytest.raises(OperationExecutionTimeoutError) as caught:
        if sat:
            _sat._solve_sat_kernel(
                cnf=CanonicalCnf(variables=("x",), clauses=((1,),)),
                timeout_ms=1000,
            )
        else:
            _smt._solve_smt_kernel(
                logic="QF_LIA",
                smtlib="(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
                timeout_ms=1000,
            )
    assert caught.value is error
