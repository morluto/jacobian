"""Finite exact LP basis search, with FLINT 0.9 owning rational elimination.

SymPy 1.14's lpmin AND linprog return infeasible points for the covering
examples in #3194/#3192. No SymPy status is used here. SoPlex (exact mode),
cddlib (GMP), and PPL offer maintained exact LP solvers, but require new native
dependencies. For this bounded domain the installed FLINT binding suffices:
enumerate bases, never implement simplex pivots or floating reconstruction.

For full row rank r, every nonempty {x>=0: Ax=b} has a basic feasible point.
A bounded optimum has a primal/dual feasible basis. An unbounded objective
has a feasible basis with a nonnegative improving fundamental ray. If every
original basis is infeasible, min t subject to Ax+b*t=b, x,t>=0 is feasible
and bounded below. Its feasible bases must contain the artificial column b.
An optimal basis has t>0 and dual y with A^T y<=0 and b^T y=t>0; -y is Farkas.
Thus the two disjoint basis families total C(n+1,r), with no pivot cycles.
Zero columns are removed reversibly; negative-cost zero columns supply rays
only AFTER a feasible point has been established.

Backend references: https://python-flint.readthedocs.io/en/latest/fmpq_mat.html
https://soplex.zib.de/ and https://pycddlib.readthedocs.io/en/stable/
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
from math import comb
from time import monotonic
from typing import Any

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.optimization._models import (
    MAX_LINEAR_PROGRAM_BASES,
    MAX_LINEAR_PROGRAM_SCALAR_UPDATES,
    StandardFormRationalLinearProgram,
    _active_equations,
    _has_trivial_inconsistent_row,
    _result_digit_bound,
)

LINEAR_PROGRAM_WALL_SECONDS = 600


@contextmanager
def linear_execution() -> Iterator[None]:
    """One cooperative deadline from entry through normalization and mapping.

    Dispatch retains this same envelope through serialization. Native callers get
    an entry-local envelope. FLINT calls are bounded small matrix primitives;
    checkpoints occur after elimination, between bases and before returning a
    certificate. This safety deadline is not an admission proof or an LP status.
    """
    execution = current_request_execution()
    if execution is None:
        with request_execution(monotonic()), linear_execution():
            yield
        return
    deadline = execution.started_at + LINEAR_PROGRAM_WALL_SECONDS
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("linear-program admission")
    yield
    request_checkpoint("linear-program result construction")


@dataclass(frozen=True)
class LinearAdmission:
    columns: tuple[int, ...]
    result_digits: int
    components: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()


def basis_bounds(variables: int, equations: int) -> tuple[int, int]:
    """Bound both searches without computing rank during admission.

    Maximize over possible ranks. Per basis reserve cubic inversion, a full
    tableau product, and linear scans/dot products. The preprocessing term covers
    two rectangular row reductions, one inverse and source-row dependence checks.
    All entries are ratios of source minors (including b and c); fraction-free
    elimination and matrix products have finite intermediate height bounded by
    (2*min(n,m)+4) times the conservative result-minor digit bound.
    """
    preprocessing = 8 * (equations + 1) ** 2 * (variables + equations + 2)
    candidates = 1
    updates = preprocessing
    for rank in range(1, min(variables, equations) + 1):
        count = comb(variables + 1, rank)
        per_basis = (
            4 * rank**3 + 2 * rank**2 * (variables + 2) + 4 * rank * (variables + 2)
        )
        candidates = max(candidates, count)
        updates = max(updates, preprocessing + count * per_basis)
    return candidates, updates


def _constraint_components(
    program: StandardFormRationalLinearProgram,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    row_columns = [
        tuple(j for j, value in enumerate(row) if value.num != 0)
        for row in program.coefficients
    ]
    column_rows: dict[int, list[int]] = {}
    for i, source_columns in enumerate(row_columns):
        for j in source_columns:
            column_rows.setdefault(j, []).append(i)
    remaining = set(column_rows)
    components = []
    while remaining:
        pending = [min(remaining)]
        rows: set[int] = set()
        columns: set[int] = set()
        while pending:
            j = pending.pop()
            if j not in remaining:
                continue
            remaining.remove(j)
            columns.add(j)
            for i in column_rows[j]:
                if i not in rows:
                    rows.add(i)
                    pending.extend(row_columns[i])
        components.append((tuple(sorted(rows)), tuple(sorted(columns))))
    return tuple(components)


def admit_linear_program(program: StandardFormRationalLinearProgram) -> LinearAdmission:
    columns = tuple(
        j
        for j in range(len(program.variables))
        if any(row[j].num != 0 for row in program.coefficients)
    )
    digits = _result_digit_bound(program)
    rows = len(_active_equations(program))
    components = _constraint_components(program)
    if _has_trivial_inconsistent_row(program):
        candidates, work = 0, 0
    else:
        bounds = [basis_bounds(len(cs), len(rs)) for rs, cs in components]
        candidates = sum(count for count, _ in bounds)
        # Source scans, subproblem projection and certificate assembly remain
        # in source coordinates; their dense work is charged independently.
        work = sum(cost for _, cost in bounds) + 16 * (len(program.rhs) + 1) * (
            len(program.variables) + 1
        )

    quantities = (
        f"normalized_columns={len(program.variables)}, active_columns={len(columns)}, "
        f"normalized_rows={len(program.rhs)}, active_rows={rows}, "
        f"basis_estimate={candidates}, basis_limit={MAX_LINEAR_PROGRAM_BASES}, "
        f"work_estimate={work}, work_limit={MAX_LINEAR_PROGRAM_SCALAR_UPDATES}, "
        f"result_digits={digits}, result_digit_limit={MAX_CANONICAL_RATIONAL_DIGITS}"
    )
    for reason, measured, limit in (
        ("result_height", digits, MAX_CANONICAL_RATIONAL_DIGITS),
        ("basis_bound", candidates, MAX_LINEAR_PROGRAM_BASES),
        ("work_bound", work, MAX_LINEAR_PROGRAM_SCALAR_UPDATES),
    ):
        if measured > limit:
            raise OperationResourceAdmissionError(
                location=("program",),
                code=f"optimization.linear.{reason}",
                message=f"Exact LP {reason} exceeded: {quantities}.",
            )
    return LinearAdmission(columns, digits, components)


def independent_rows(a: Any, b: Any) -> tuple[tuple[int, ...], Any | None]:
    """Select source rows, or produce a source-coordinate dependence witness."""
    from flint import fmpq_mat

    reduced, rank = a.transpose().rref()
    request_checkpoint("linear-program row reduction")
    indices = tuple(
        next(j for j in range(a.nrows()) if reduced[i, j]) for i in range(rank)
    )
    selected = fmpq_mat([[a[i, j] for j in range(a.ncols())] for i in indices])
    reduced_rows, _ = selected.rref()
    pivots = tuple(
        next(j for j in range(a.ncols()) if reduced_rows[i, j]) for i in range(rank)
    )
    square = fmpq_mat([[a[i, j] for j in pivots] for i in indices])
    weights = (
        fmpq_mat([[a[i, j] for j in pivots] for i in range(a.nrows())]) * square.inv()
    )
    residual = b - weights * fmpq_mat([[b[i, 0]] for i in indices])
    request_checkpoint("linear-program row consistency")
    for i in range(a.nrows()):
        if residual[i, 0]:
            witness = fmpq_mat(a.nrows(), 1)
            sign = 1 if residual[i, 0] > 0 else -1
            witness[i, 0] = -sign
            for j, source in enumerate(indices):
                witness[source, 0] += sign * weights[i, j]
            return indices, witness
    return indices, None


def search_bases(
    a: Any, b: Any, c: Any, *, artificial: bool = False
) -> tuple[Any, Any, Any | None] | None:
    """Return a primal/dual pair or a primal/ray pair from one finite family."""
    from flint import fmpq_mat

    rows, columns = a.nrows(), a.ncols()
    family = (
        ((*basis, columns - 1) for basis in combinations(range(columns - 1), rows - 1))
        if artificial
        else combinations(range(columns), rows)
    )
    for basis in family:
        request_checkpoint("linear-program basis search")
        square = fmpq_mat([[a[i, j] for j in basis] for i in range(rows)])
        try:
            inverse = square.inv()
        except ZeroDivisionError:  # FLINT's documented singular-matrix outcome.
            continue
        basic_point = inverse * b
        if any(basic_point[i, 0] < 0 for i in range(rows)):
            continue
        point = fmpq_mat(columns, 1)
        for i, j in enumerate(basis):
            point[j, 0] = basic_point[i, 0]
        dual = fmpq_mat([[c[0, j] for j in basis]]) * inverse
        slacks = c - dual * a
        if all(slacks[0, j] >= 0 for j in range(columns)):
            return point, dual.transpose(), None
        if artificial:
            continue  # min t >= 0 cannot be unbounded below.
        tableau = inverse * a
        for j in range(columns):
            if slacks[0, j] < 0 and all(tableau[i, j] <= 0 for i in range(rows)):
                ray = fmpq_mat(columns, 1)
                ray[j, 0] = 1
                for i, k in enumerate(basis):
                    ray[k, 0] = -tableau[i, j]
                return point, dual.transpose(), ray
    return None
