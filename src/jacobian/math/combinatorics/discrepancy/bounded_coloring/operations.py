"""Source-aligned bounded discrepancy decision and exact SAT-ledger replay."""

from itertools import groupby
from time import monotonic

from jacobian._execution import (
    current_request_execution,
    request_checkpoint,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.discrepancy._models import FiniteSetSystem
from jacobian.math.combinatorics.discrepancy.bounded_coloring._models import (
    BoundedColoringBudget,
    BoundedColoringBudgetExceeded,
    BoundedColoringExecutionFailed,
    BoundedColoringOutcome,
    BoundedColoringResult,
    SatisfiableBoundedColoring,
    UnsatisfiableBoundedColoring,
    _require_bounds,
)
from jacobian.math.combinatorics.discrepancy.bounded_coloring._process import run_solver
from jacobian.math.combinatorics.discrepancy.bounded_coloring._z3 import (
    BackendReply,
    Constraints,
)

__all__ = ["decide"]


class _WallBudgetExceededError(Exception):
    """The operation budget expired without exhausting a caller deadline."""


def _checkpoint(deadline: float, stage: str) -> None:
    request_checkpoint(stage)
    if monotonic() >= deadline:
        raise _WallBudgetExceededError


def _admit(source: FiniteSetSystem, bounds: tuple[int, ...]) -> Constraints:
    """Presolve duplicate supports and tautologies, admitting the full carrier.

    The canonical carrier has n<=64, m<=1000 and I<=64000. Bound widths are
    <=7 bits; a SAT ledger has exactly m integers in [-64,64] and n signs.
    After presolve there are <=2000 unit-weight PB constraints with <=128000
    literal occurrences. Original source plus bounds/ledger has fewer than
    256 KiB of decimal JSON, derived from these mathematical counts. Formula
    construction is O((I+m) log(m+1)); search has a separate enforced Z3 work cap,
    512-MiB solver memory cap, and killable shared deadline/OS containment.

    If q is the number of positive signs on a set of size s, its signed sum
    is 2q-s. Thus ceil((s-b)/2)<=q<=floor((s+b)/2) is exactly the requested
    absolute bound, including parity-impossible singleton constraints.
    """
    try:
        _require_bounds(source, bounds)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("absolute_bounds",),
            code="discrepancy_theory.bound_axis",
            message=str(exc),
        ) from exc
    constraints: list[tuple[tuple[int, ...], int, int]] = []
    rows = sorted(zip(source.sets, bounds, strict=True))
    for subset, duplicates in groupby(rows, key=lambda row: row[0]):
        bound = min(row[1] for row in duplicates)
        size = len(subset)
        if bound == size:
            continue
        constraints.append((subset, (size - bound + 1) // 2, (size + bound) // 2))
    return tuple(constraints)


def _outcome(
    source: FiniteSetSystem, bounds: tuple[int, ...], reply: BackendReply
) -> BoundedColoringOutcome:
    """Promote only a complete, exactly replayed backend SAT assignment."""
    if reply["status"] == "SATISFIABLE":
        coloring = reply["coloring"]
        if (
            coloring is None
            or len(coloring) != source.ground_set_size
            or any(type(value) is not int or value not in (-1, 1) for value in coloring)
        ):
            return BoundedColoringExecutionFailed()
        signed_sums = tuple(
            sum(coloring[index] for index in subset) for subset in source.sets
        )
        if any(
            abs(total) > bound for total, bound in zip(signed_sums, bounds, strict=True)
        ):
            return BoundedColoringExecutionFailed()
        return SatisfiableBoundedColoring(coloring=coloring, signed_sums=signed_sums)
    if reply["coloring"] is not None:
        return BoundedColoringExecutionFailed()
    if reply["status"] == "UNSATISFIABLE":
        return UnsatisfiableBoundedColoring()
    if reply["status"] == "BUDGET_EXCEEDED":
        return BoundedColoringBudgetExceeded()
    return BoundedColoringExecutionFailed()


def decide(
    set_system: FiniteSetSystem,
    absolute_bounds: tuple[int, ...],
    resource_budget: BoundedColoringBudget | None = None,
) -> BoundedColoringResult:
    """Decide whether one signed coloring satisfies every indexed set bound."""
    budget = resource_budget or BoundedColoringBudget()
    execution = current_request_execution()
    deadline = (
        execution.started_at if execution is not None else monotonic()
    ) + budget.wall_seconds
    if execution is not None and execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    # Keep the caller envelope intact: its timeout/cancellation are operational
    # errors; this operation's explicit wall budget has a claim-free outcome.
    try:
        _checkpoint(deadline, "before discrepancy admission")
        constraints = _admit(set_system, absolute_bounds)
        if constraints:
            reply = run_solver(
                set_system.ground_set_size,
                constraints,
                budget.solver_work_limit,
                deadline,
            )
        else:
            reply = {
                "status": "SATISFIABLE",
                "coloring": (1,) * set_system.ground_set_size,
            }
        _checkpoint(deadline, "before discrepancy witness replay")
        outcome = _outcome(set_system, absolute_bounds, reply)
        _checkpoint(deadline, "before discrepancy result construction")
        result = BoundedColoringResult(
            set_system=set_system, absolute_bounds=absolute_bounds, outcome=outcome
        )
        _checkpoint(deadline, "after discrepancy result construction")
        return result
    except _WallBudgetExceededError:
        # Only bounded source copying remains; no solver evidence is promoted.
        result = BoundedColoringResult(
            set_system=set_system,
            absolute_bounds=absolute_bounds,
            outcome=BoundedColoringBudgetExceeded(),
        )
        request_checkpoint("after discrepancy budget result construction")
        return result
