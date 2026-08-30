"""Exact bounded kernel for clause-constrained rational flats."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from math import factorial, gcd, lcm

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    current_request_execution,
    request_cancelled,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.math.combinatorics.matroids.rational_flats._models import (
    MAX_RATIONAL_FLAT_GROUP_ORDER,
    MAX_RATIONAL_FLAT_RESULT_ORBITS,
    ClauseConstrainedRationalFlatClassification,
    ClauseConstrainedRationalFlatProblem,
    RationalFlatIncompleteReason,
    RationalFlatOrbitRepresentative,
    _validation_error,
)
from jacobian.math.matrices.values import (
    MAX_MATRIX_SCALAR_DIGITS,
    RationalVectorSpaceBasis,
    SparseRationalMatrix,
    rational_vector_space_basis_from_fractions,
)

MAX_RATIONAL_FLAT_INPUT_COMPONENT_DIGITS = 256
MAX_RATIONAL_FLAT_SEARCH_STATE_ORBITS = 100_000
MAX_RATIONAL_FLAT_SEARCH_WORK = 5_000_000_000
MAX_RATIONAL_FLAT_ORBIT_CACHE_ENTRIES = 500_000
_RESULT_ENVELOPE_RESERVE_BYTES = 16_384

type RationalRow = tuple[Fraction, ...]
type IntegerRow = tuple[int, ...]
type ClosedCandidateSet = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _RationalFlatPlan:
    deadline: float | None
    candidate_rows: tuple[IntegerRow, ...]
    forbidden_rows: tuple[IntegerRow, ...]
    candidate_permutations: tuple[tuple[int, ...], ...]
    symmetry_group_order: int
    state_orbit_limit: int
    result_orbit_limit: int
    search_work_limit: int
    linear_algebra_chunk_cost: int


@dataclass(frozen=True, slots=True)
class _FlatState:
    closed_candidates: ClosedCandidateSet
    row_space_basis: tuple[RationalRow, ...]
    rank: int


class _SearchStoppedError(Exception):
    def __init__(
        self,
        reason: RationalFlatIncompleteReason,
        *,
        visited_count: int = 0,
        consumed_work: int = 0,
    ) -> None:
        self.reason = reason
        self.visited_count = visited_count
        self.consumed_work = consumed_work
        super().__init__(reason)


@dataclass(slots=True)
class _WorkLedger:
    limit: int
    linear_algebra_chunk_cost: int
    deadline: float | None
    consumed: int = 0
    state_orbit_count: int = 0

    def charge(self, units: int) -> None:
        units = max(units, 1)
        if self.consumed + units > self.limit:
            raise _SearchStoppedError(
                "SEARCH_WORK_LIMIT",
                visited_count=self.state_orbit_count,
                consumed_work=self.consumed,
            )
        self.consumed += units

    def charge_linear_algebra(self, scalar_operations: int) -> None:
        self.charge(scalar_operations * self.linear_algebra_chunk_cost)


def _require_execution_active(deadline: float | None, phase: str) -> None:
    if request_cancelled():
        raise OperationExecutionCancelledError(f"request cancelled {phase}")
    if deadline is not None and deadline <= time.monotonic():
        raise OperationExecutionTimeoutError(f"request deadline expired {phase}")


def _dense_fraction_rows(matrix: SparseRationalMatrix) -> tuple[RationalRow, ...]:
    rows = [
        [Fraction(0) for _ in range(matrix.column_count)]
        for _ in range(matrix.row_count)
    ]
    for entry in matrix.entries:
        rows[entry.row][entry.column] = entry.value.as_fraction()
    return tuple(tuple(row) for row in rows)


def _primitive_integer_row(row: RationalRow) -> IntegerRow:
    """Return the canonical primitive integer representative of one QQ line."""

    if not any(row):
        return tuple(0 for _ in row)
    denominator = lcm(*(value.denominator for value in row))
    integers = [value.numerator * (denominator // value.denominator) for value in row]
    content = gcd(*(abs(value) for value in integers))
    primitive = [value // content for value in integers]
    if next(value for value in primitive if value) < 0:
        primitive = [-value for value in primitive]
    return tuple(primitive)


def _permuted_projective_row(
    row: IntegerRow,
    coordinate_permutation: tuple[int, ...],
) -> IntegerRow:
    image = [0] * len(row)
    for source, target in enumerate(coordinate_permutation):
        image[target] = row[source]
    if any(image) and next(value for value in image if value) < 0:
        image = [-value for value in image]
    return tuple(image)


def _require_component_digits(problem: ClauseConstrainedRationalFlatProblem) -> None:
    for label, matrix in (
        ("candidate", problem.candidates.vectors),
        ("forbidden", problem.forbidden_vectors),
    ):
        for entry in matrix.entries:
            if any(
                len(component.lstrip("-")) > MAX_RATIONAL_FLAT_INPUT_COMPONENT_DIGITS
                for component in (entry.value.num, entry.value.den)
            ):
                raise _validation_error(
                    "input_component_bound",
                    f"{label} rational components may contain at most "
                    f"{MAX_RATIONAL_FLAT_INPUT_COMPONENT_DIGITS} decimal digits",
                )


def _minor_digit_bound(
    candidate_rows: tuple[IntegerRow, ...],
    maximum_rank: int,
) -> int:
    """Bound RREF numerator and denominator minors by rowwise Hadamard data."""

    if maximum_rank == 0:
        return 1
    row_digits = sorted(
        (
            max(1, max(len(str(abs(value))) for value in row))
            for row in candidate_rows
            if any(row)
        ),
        reverse=True,
    )
    rank = min(maximum_rank, len(row_digits))
    if rank == 0:
        return 1
    # Leibniz expansion bounds every rank minor by rank! times the product of
    # one entry bound from each selected row.  This also bounds RREF ratios of
    # adjacent maximal minors and the canonical annihilator coordinates.
    return sum(row_digits[:rank]) + len(str(factorial(rank))) + 1


def _largest_row_component_digits(rows: tuple[IntegerRow, ...]) -> int:
    return max(
        (max(1, max(len(str(abs(value))) for value in row)) for row in rows),
        default=1,
    )


def _require_symmetry_compatibility(
    problem: ClauseConstrainedRationalFlatProblem,
    *,
    candidate_rows: tuple[IntegerRow, ...],
    forbidden_rows: tuple[IntegerRow, ...],
) -> tuple[tuple[int, ...], ...]:
    clauses = set(problem.clauses)
    forbidden_lines = set(forbidden_rows)
    candidate_permutations: list[tuple[int, ...]] = []
    for generator in problem.symmetry_generators:
        coordinate_permutation = tuple(generator.coordinate_permutation)
        candidate_permutation = tuple(generator.candidate_permutation)
        for source, target in enumerate(candidate_permutation):
            if (
                _permuted_projective_row(candidate_rows[source], coordinate_permutation)
                != candidate_rows[target]
            ):
                raise _validation_error(
                    "candidate_symmetry",
                    "each paired generator must send every candidate row to its "
                    "declared projective image",
                )
        transformed_clauses = {
            tuple(sorted(candidate_permutation[index] for index in clause))
            for clause in clauses
        }
        if transformed_clauses != clauses:
            raise _validation_error(
                "clause_symmetry",
                "each candidate generator must preserve the complete clause family",
            )
        transformed_forbidden = {
            _permuted_projective_row(row, coordinate_permutation)
            for row in forbidden_lines
        }
        if transformed_forbidden != forbidden_lines:
            raise _validation_error(
                "forbidden_symmetry",
                "each coordinate generator must preserve the forbidden projective-row set",
            )
        candidate_permutations.append(candidate_permutation)
    return tuple(candidate_permutations)


def _compose_permutations(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    """Return ``left ∘ right`` in array form."""

    return tuple(left[right[index]] for index in range(len(left)))


def _paired_group_order(
    problem: ClauseConstrainedRationalFlatProblem,
    *,
    deadline: float | None,
) -> int:
    """Enumerate the admitted paired action, stopping at the order boundary.

    The repository's finite-action owner uses the same bounded closure pattern.
    Here the acted-on set can contain 16 coordinate positions plus 128 labelled
    candidates, so its narrower public carrier cannot represent this paired
    action unchanged.  Enumerating at most 10,001 elements makes the backend
    work a consequence of the public order, degree, and generator bounds and
    rejects an oversized group as soon as that fact has been established.
    """

    if not problem.symmetry_generators:
        return 1

    ambient_dimension = len(problem.candidates.coordinate_axis)
    candidate_count = problem.candidates.vector_count
    paired_generators = tuple(
        (
            *generator.coordinate_permutation,
            *(ambient_dimension + image for image in generator.candidate_permutation),
        )
        for generator in problem.symmetry_generators
    )
    degree = ambient_dimension + candidate_count
    identity = tuple(range(degree))
    seen = {identity}
    pending = deque((identity,))
    while pending:
        _require_execution_active(deadline, "during finite symmetry recognition")
        current = pending.popleft()
        for generator in paired_generators:
            image = _compose_permutations(generator, current)
            if image in seen:
                continue
            if len(seen) >= MAX_RATIONAL_FLAT_GROUP_ORDER:
                raise _validation_error(
                    "symmetry_group_order_bound",
                    "the paired symmetry group exceeds the admitted order bound of "
                    f"{MAX_RATIONAL_FLAT_GROUP_ORDER}",
                )
            seen.add(image)
            pending.append(image)
    return len(seen)


def _result_orbit_limit(
    problem: ClauseConstrainedRationalFlatProblem,
    *,
    source_bytes: int,
    minor_digits: int,
) -> int:
    output_limit = CanonicalLimits().max_output_bytes
    available = output_limit - source_bytes - _RESULT_ENVELOPE_RESERVE_BYTES
    if available < 0:
        raise _validation_error(
            "result_size_bound",
            "the retained rational-flat problem leaves no room for a result",
        )
    ambient_dimension = len(problem.candidates.coordinate_axis)
    candidate_count = problem.candidates.vector_count
    scalar_bytes = 2 * minor_digits + 96
    per_representative = (
        4_096
        + candidate_count * (len(str(max(candidate_count, 1))) + 3)
        + 2 * ambient_dimension * ambient_dimension * scalar_bytes
    )
    return min(
        MAX_RATIONAL_FLAT_RESULT_ORBITS,
        available // max(per_representative, 1),
    )


def _admit_problem(problem: ClauseConstrainedRationalFlatProblem) -> _RationalFlatPlan:
    """Build one pre-search plan for all exact work and result obligations."""

    execution = current_request_execution()
    deadline = execution.deadline if execution is not None else None
    _require_execution_active(deadline, "before rational-flat admission")
    _require_component_digits(problem)
    candidate_rows = tuple(
        _primitive_integer_row(row)
        for row in _dense_fraction_rows(problem.candidates.vectors)
    )
    forbidden_rows = tuple(
        _primitive_integer_row(row)
        for row in _dense_fraction_rows(problem.forbidden_vectors)
    )
    _require_execution_active(deadline, "after rational-row normalization")
    minor_digits = _minor_digit_bound(candidate_rows, problem.maximum_rank)
    if minor_digits > MAX_MATRIX_SCALAR_DIGITS:
        raise _validation_error(
            "rref_component_bound",
            "candidate heights can make canonical RREF or annihilator components "
            f"exceed {MAX_MATRIX_SCALAR_DIGITS} decimal digits",
        )
    linear_algebra_digits = max(
        minor_digits,
        _largest_row_component_digits(candidate_rows),
        _largest_row_component_digits(forbidden_rows),
    )
    linear_algebra_chunks = (linear_algebra_digits + 31) // 32
    linear_algebra_chunk_cost = linear_algebra_chunks * linear_algebra_chunks
    candidate_permutations = _require_symmetry_compatibility(
        problem,
        candidate_rows=candidate_rows,
        forbidden_rows=forbidden_rows,
    )
    _require_execution_active(deadline, "before finite symmetry recognition")
    group_order = _paired_group_order(problem, deadline=deadline)
    _require_execution_active(deadline, "after finite symmetry recognition")
    try:
        source_bytes = len(encode_strict_json(problem.model_dump(mode="json")))
    except CanonicalizationError:
        raise _validation_error(
            "result_size_bound",
            "the retained rational-flat problem exceeds the canonical output envelope",
        ) from None
    result_orbit_limit = _result_orbit_limit(
        problem,
        source_bytes=source_bytes,
        minor_digits=minor_digits,
    )
    ambient_dimension = len(problem.candidates.coordinate_axis)
    candidate_count = problem.candidates.vector_count
    forbidden_count = problem.forbidden_vectors.row_count
    maximum_rank = problem.maximum_rank
    per_state_work = max(
        1,
        (candidate_count + forbidden_count)
        * max(maximum_rank, 1)
        * ambient_dimension
        * linear_algebra_chunk_cost
        + group_order * max(len(candidate_permutations), 1) * max(candidate_count, 1),
    )
    state_orbit_limit = max(
        1,
        min(
            MAX_RATIONAL_FLAT_SEARCH_STATE_ORBITS,
            MAX_RATIONAL_FLAT_SEARCH_WORK // per_state_work,
        ),
    )
    return _RationalFlatPlan(
        deadline=deadline,
        candidate_rows=candidate_rows,
        forbidden_rows=forbidden_rows,
        candidate_permutations=candidate_permutations,
        symmetry_group_order=group_order,
        state_orbit_limit=state_orbit_limit,
        result_orbit_limit=result_orbit_limit,
        search_work_limit=MAX_RATIONAL_FLAT_SEARCH_WORK,
        linear_algebra_chunk_cost=linear_algebra_chunk_cost,
    )


def _rref_basis(
    rows: tuple[RationalRow | IntegerRow, ...],
    *,
    ambient_dimension: int,
    ledger: _WorkLedger,
) -> tuple[RationalRow, ...]:
    if not rows:
        return ()
    _require_execution_active(ledger.deadline, "before exact rational row reduction")
    ledger.charge_linear_algebra(
        len(rows) * ambient_dimension * max(1, min(len(rows), ambient_dimension))
    )
    from flint import fmpq, fmpq_mat

    reduced, rank = fmpq_mat(
        [
            [
                fmpq(value.numerator, value.denominator)
                if isinstance(value, Fraction)
                else fmpq(value)
                for value in row
            ]
            for row in rows
        ]
    ).rref()
    _require_execution_active(ledger.deadline, "after exact rational row reduction")
    return tuple(
        tuple(
            Fraction(int(reduced[row, column].p), int(reduced[row, column].q))
            for column in range(ambient_dimension)
        )
        for row in range(int(rank))
    )


def _pivot_columns(basis: tuple[RationalRow, ...]) -> tuple[int, ...]:
    return tuple(
        next(index for index, value in enumerate(row) if value) for row in basis
    )


def _row_is_in_span(row: IntegerRow, basis: tuple[RationalRow, ...]) -> bool:
    residual = [Fraction(value) for value in row]
    for basis_row, pivot in zip(basis, _pivot_columns(basis), strict=True):
        multiplier = residual[pivot]
        if multiplier:
            residual = [
                value - multiplier * basis_value
                for value, basis_value in zip(residual, basis_row, strict=True)
            ]
    return not any(residual)


def _closed_candidates(
    candidate_rows: tuple[IntegerRow, ...],
    basis: tuple[RationalRow, ...],
    ledger: _WorkLedger,
) -> ClosedCandidateSet:
    ambient_dimension = (
        len(candidate_rows[0]) if candidate_rows else (len(basis[0]) if basis else 1)
    )
    ledger.charge_linear_algebra(
        max(len(candidate_rows), 1) * max(len(basis), 1) * max(ambient_dimension, 1)
    )
    return tuple(
        index for index, row in enumerate(candidate_rows) if _row_is_in_span(row, basis)
    )


class _SubsetOrbitCanonicalizer:
    def __init__(self, generators: tuple[tuple[int, ...], ...]) -> None:
        self._generators = generators
        self._cache: dict[ClosedCandidateSet, tuple[ClosedCandidateSet, int]] = {}

    def canonicalize(
        self,
        subset: ClosedCandidateSet,
        ledger: _WorkLedger,
    ) -> tuple[ClosedCandidateSet, int]:
        cached = self._cache.get(subset)
        if cached is not None:
            return cached
        if not self._generators:
            result = (subset, 1)
            self._cache[subset] = result
            return result

        seen = {subset}
        pending = deque((subset,))
        while pending:
            _require_execution_active(
                ledger.deadline,
                "during rational-flat orbit canonicalization",
            )
            current = pending.popleft()
            for generator in self._generators:
                ledger.charge(max(len(current), 1))
                image = tuple(sorted(generator[index] for index in current))
                if image not in seen:
                    seen.add(image)
                    pending.append(image)
        canonical = min(seen)
        result = (canonical, len(seen))
        if len(self._cache) + len(seen) <= MAX_RATIONAL_FLAT_ORBIT_CACHE_ENTRIES:
            for image in seen:
                self._cache[image] = result
        return result


def _annihilator_basis(
    basis: tuple[RationalRow, ...],
    *,
    ambient_dimension: int,
) -> tuple[RationalRow, ...]:
    pivots = _pivot_columns(basis)
    pivot_set = set(pivots)
    vectors: list[RationalRow] = []
    for free_column in range(ambient_dimension):
        if free_column in pivot_set:
            continue
        vector = [Fraction(0)] * ambient_dimension
        vector[free_column] = Fraction(1)
        for row_index, pivot in enumerate(pivots):
            vector[pivot] = -basis[row_index][free_column]
        vectors.append(tuple(vector))
    return tuple(vectors)


def _basis_value(
    basis: tuple[RationalRow, ...],
    *,
    ambient_dimension: int,
) -> RationalVectorSpaceBasis:
    return rational_vector_space_basis_from_fractions(
        basis,
        ambient_dimension=ambient_dimension,
    )


def _state_from_closed(
    closed: ClosedCandidateSet,
    *,
    plan: _RationalFlatPlan,
    ambient_dimension: int,
    ledger: _WorkLedger,
    state_cache: dict[ClosedCandidateSet, _FlatState],
) -> _FlatState:
    cached = state_cache.get(closed)
    if cached is not None:
        return cached
    basis = _rref_basis(
        tuple(plan.candidate_rows[index] for index in closed),
        ambient_dimension=ambient_dimension,
        ledger=ledger,
    )
    state = _FlatState(
        closed_candidates=closed,
        row_space_basis=basis,
        rank=len(basis),
    )
    state_cache[closed] = state
    return state


def _canonical_closure(
    rows: tuple[RationalRow | IntegerRow, ...],
    *,
    plan: _RationalFlatPlan,
    ambient_dimension: int,
    ledger: _WorkLedger,
    canonicalizer: _SubsetOrbitCanonicalizer,
) -> ClosedCandidateSet:
    basis = _rref_basis(
        rows,
        ambient_dimension=ambient_dimension,
        ledger=ledger,
    )
    closed = _closed_candidates(plan.candidate_rows, basis, ledger)
    canonical, _orbit_size = canonicalizer.canonicalize(closed, ledger)
    return canonical


def _contains_forbidden_row(
    state: _FlatState,
    *,
    plan: _RationalFlatPlan,
    ambient_dimension: int,
    ledger: _WorkLedger,
) -> bool:
    ledger.charge_linear_algebra(
        max(len(plan.forbidden_rows), 1) * max(state.rank, 1) * ambient_dimension
    )
    return any(
        _row_is_in_span(row, state.row_space_basis) for row in plan.forbidden_rows
    )


def _branch_candidates(
    problem: ClauseConstrainedRationalFlatProblem,
    closed: ClosedCandidateSet,
) -> tuple[int, ...]:
    closed_set = set(closed)
    unmet_clause = next(
        (clause for clause in problem.clauses if closed_set.isdisjoint(clause)),
        None,
    )
    if unmet_clause is not None:
        return tuple(index for index in unmet_clause if index not in closed_set)
    return tuple(
        index
        for index in range(problem.candidates.vector_count)
        if index not in closed_set
    )


def _clauses_are_satisfied(
    problem: ClauseConstrainedRationalFlatProblem,
    closed: ClosedCandidateSet,
) -> bool:
    closed_set = set(closed)
    return all(not closed_set.isdisjoint(clause) for clause in problem.clauses)


def _search_satisfying_states(
    problem: ClauseConstrainedRationalFlatProblem,
    plan: _RationalFlatPlan,
) -> tuple[
    dict[ClosedCandidateSet, _FlatState],
    set[ClosedCandidateSet],
    _WorkLedger,
    _SubsetOrbitCanonicalizer,
]:
    ambient_dimension = len(problem.candidates.coordinate_axis)
    ledger = _WorkLedger(
        limit=plan.search_work_limit,
        linear_algebra_chunk_cost=plan.linear_algebra_chunk_cost,
        deadline=plan.deadline,
    )
    canonicalizer = _SubsetOrbitCanonicalizer(plan.candidate_permutations)
    state_cache: dict[ClosedCandidateSet, _FlatState] = {}
    visited: set[ClosedCandidateSet] = set()
    satisfying: dict[ClosedCandidateSet, _FlatState] = {}
    initial = _canonical_closure(
        (),
        plan=plan,
        ambient_dimension=ambient_dimension,
        ledger=ledger,
        canonicalizer=canonicalizer,
    )
    pending = [initial]
    queued = {initial}
    while pending:
        _require_execution_active(plan.deadline, "during rational-flat search")
        closed = pending.pop()
        queued.discard(closed)
        if closed in visited:
            continue
        if len(visited) >= plan.state_orbit_limit:
            raise _SearchStoppedError(
                "STATE_ORBIT_LIMIT",
                visited_count=len(visited),
                consumed_work=ledger.consumed,
            )
        visited.add(closed)
        ledger.state_orbit_count = len(visited)
        state = _state_from_closed(
            closed,
            plan=plan,
            ambient_dimension=ambient_dimension,
            ledger=ledger,
            state_cache=state_cache,
        )
        if state.rank > problem.maximum_rank or _contains_forbidden_row(
            state,
            plan=plan,
            ambient_dimension=ambient_dimension,
            ledger=ledger,
        ):
            continue
        clauses_satisfied = _clauses_are_satisfied(problem, closed)
        if clauses_satisfied and state.rank >= problem.minimum_rank:
            if closed not in satisfying and len(satisfying) >= plan.result_orbit_limit:
                raise _SearchStoppedError(
                    "RESULT_ORBIT_LIMIT",
                    visited_count=len(visited),
                    consumed_work=ledger.consumed,
                )
            satisfying[closed] = state
        if state.rank == problem.maximum_rank:
            continue
        child_keys = {
            _canonical_closure(
                (*state.row_space_basis, plan.candidate_rows[index]),
                plan=plan,
                ambient_dimension=ambient_dimension,
                ledger=ledger,
                canonicalizer=canonicalizer,
            )
            for index in _branch_candidates(problem, closed)
        }
        new_children = tuple(sorted(child_keys.difference(visited, queued)))
        if len(visited) + len(queued) + len(new_children) > plan.state_orbit_limit:
            raise _SearchStoppedError(
                "STATE_ORBIT_LIMIT",
                visited_count=len(visited),
                consumed_work=ledger.consumed,
            )
        queued.update(new_children)
        pending.extend(reversed(new_children))
    return satisfying, visited, ledger, canonicalizer


def _representatives_from_states(
    satisfying: dict[ClosedCandidateSet, _FlatState],
    *,
    plan: _RationalFlatPlan,
    ambient_dimension: int,
    ledger: _WorkLedger,
    canonicalizer: _SubsetOrbitCanonicalizer,
) -> tuple[RationalFlatOrbitRepresentative, ...]:
    representatives: list[RationalFlatOrbitRepresentative] = []
    for closed, state in sorted(satisfying.items()):
        _require_execution_active(
            ledger.deadline,
            "during rational-flat representative construction",
        )
        _canonical, orbit_size = canonicalizer.canonicalize(closed, ledger)
        ledger.charge_linear_algebra(ambient_dimension * ambient_dimension)
        representatives.append(
            RationalFlatOrbitRepresentative._from_kernel(
                closed_candidate_indices=closed,
                rank=state.rank,
                row_space_basis=_basis_value(
                    state.row_space_basis,
                    ambient_dimension=ambient_dimension,
                ),
                annihilator_basis=_basis_value(
                    _annihilator_basis(
                        state.row_space_basis,
                        ambient_dimension=ambient_dimension,
                    ),
                    ambient_dimension=ambient_dimension,
                ),
                orbit_size=orbit_size,
                stabilizer_order=plan.symmetry_group_order // orbit_size,
            )
        )
    return tuple(representatives)


def _incomplete_result(
    problem: ClauseConstrainedRationalFlatProblem,
    plan: _RationalFlatPlan,
    *,
    reason: RationalFlatIncompleteReason,
    visited_count: int,
    consumed_search_work: int,
) -> ClauseConstrainedRationalFlatClassification:
    return ClauseConstrainedRationalFlatClassification._incomplete_from_kernel(
        problem=problem,
        symmetry_group_order=plan.symmetry_group_order,
        reason=reason,
        explored_state_orbit_count=visited_count,
        state_orbit_limit=plan.state_orbit_limit,
        result_orbit_limit=plan.result_orbit_limit,
        consumed_search_work=consumed_search_work,
        search_work_limit=plan.search_work_limit,
    )


def _classify(
    problem: ClauseConstrainedRationalFlatProblem,
    plan: _RationalFlatPlan,
) -> ClauseConstrainedRationalFlatClassification:
    try:
        satisfying, _visited, ledger, canonicalizer = _search_satisfying_states(
            problem, plan
        )
        ambient_dimension = len(problem.candidates.coordinate_axis)
        representatives = _representatives_from_states(
            satisfying,
            plan=plan,
            ambient_dimension=ambient_dimension,
            ledger=ledger,
            canonicalizer=canonicalizer,
        )
    except _SearchStoppedError as stopped:
        # A stopped search has not established a mathematical family.  The
        # bounded outcome therefore discards partial representatives.
        return _incomplete_result(
            problem,
            plan,
            reason=stopped.reason,
            visited_count=stopped.visited_count,
            consumed_search_work=stopped.consumed_work,
        )
    result = ClauseConstrainedRationalFlatClassification._complete_from_kernel(
        problem=problem,
        symmetry_group_order=plan.symmetry_group_order,
        representatives=representatives,
    )
    _require_execution_active(plan.deadline, "before rational-flat result delivery")
    return result


def classify_clause_constrained_rational_flats_kernel(
    problem: ClauseConstrainedRationalFlatProblem,
) -> ClauseConstrainedRationalFlatClassification:
    """Return the complete satisfying flat orbits when the bounded search exhausts."""

    plan = _admit_problem(problem)
    return _classify(problem, plan)


__all__ = ["classify_clause_constrained_rational_flats_kernel"]
