"""Maintained Z3 adapter for one admitted vector of cardinality bounds."""

from time import monotonic
from typing import Literal, TypedDict

from jacobian._execution import OperationExecutionTimeoutError

Constraints = tuple[tuple[tuple[int, ...], int, int], ...]
BackendStatus = Literal[
    "SATISFIABLE", "UNSATISFIABLE", "BUDGET_EXCEEDED", "EXECUTION_FAILED"
]


class BackendReply(TypedDict):
    status: BackendStatus
    coloring: tuple[int, ...] | None


def _deadline_reply(caller_limited: bool, message: str) -> BackendReply:
    if caller_limited:
        raise OperationExecutionTimeoutError(message)
    return {"status": "BUDGET_EXCEEDED", "coloring": None}


def solve(
    variable_count: int,
    constraints: Constraints,
    work_limit: int,
    deadline: float,
    *,
    caller_limited: bool = False,
) -> BackendReply:
    """Decide exact unit-weight PB constraints using Z3 5.1's resource limit.

    One check is allowed. Z3 rlimit bounds internal proof work; timeout and
    max_memory additionally bound the native backend. Process containment is
    owned by the caller, so a failed/unknown native solve remains claim-free.
    """
    try:
        import z3
    except (ImportError, OSError):
        return {"status": "EXECUTION_FAILED", "coloring": None}
    try:
        solver = z3.Solver()
        solver.set(rlimit=work_limit, max_memory=512)
        bits = [z3.Bool(f"bounded_color_{index}") for index in range(variable_count)]
        for subset, lower, upper in constraints:
            if monotonic() >= deadline:
                return _deadline_reply(
                    caller_limited,
                    "discrepancy decision deadline expired while encoding constraints",
                )
            weighted = [(bits[index], 1) for index in subset]
            solver.add(z3.PbGe(weighted, lower), z3.PbLe(weighted, upper))
        # All absolute-sum bounds are invariant under global sign reversal.
        if variable_count:
            solver.add(bits[0])
        remaining_ms = int((deadline - monotonic()) * 1000)
        if remaining_ms <= 0:
            return _deadline_reply(
                caller_limited,
                "discrepancy decision deadline expired before the Z3 check",
            )
        solver.set(timeout=remaining_ms)
        status = solver.check()
        if monotonic() >= deadline:
            return _deadline_reply(
                caller_limited,
                "discrepancy decision deadline expired during the Z3 check",
            )
        if status == z3.unsat:
            return {"status": "UNSATISFIABLE", "coloring": None}
        if status == z3.sat:
            model = solver.model()
            values = tuple(model.eval(bit, model_completion=True) for bit in bits)
            if any(not (z3.is_true(value) or z3.is_false(value)) for value in values):
                return {"status": "EXECUTION_FAILED", "coloring": None}
            return {
                "status": "SATISFIABLE",
                "coloring": tuple(1 if z3.is_true(value) else -1 for value in values),
            }
        reason = solver.reason_unknown().lower()
        if any(
            word in reason for word in ("timeout", "resource", "memory", "canceled")
        ):
            if caller_limited and "timeout" in reason:
                raise OperationExecutionTimeoutError(
                    "discrepancy decision deadline expired during the Z3 check"
                )
            return {"status": "BUDGET_EXCEEDED", "coloring": None}
        return {"status": "EXECUTION_FAILED", "coloring": None}
    except (z3.Z3Exception, MemoryError):
        return {"status": "EXECUTION_FAILED", "coloring": None}
