"""Exact bounded kernel for clause-constrained prime-field flats."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
)
from jacobian.math.combinatorics.matroids.prime_field_flats._models import (
    MAX_PRIME_FIELD_FLAT_CLAUSE_MEMBERSHIPS,
    MAX_PRIME_FIELD_FLAT_GROUP_ORDER,
    MAX_PRIME_FIELD_FLAT_RESULT_ORBITS,
    ClauseConstrainedPrimeFieldFlatClassification,
    ClauseConstrainedPrimeFieldFlatProblem,
    PrimeFieldFlatIncompleteReason,
    PrimeFieldFlatOrbitRepresentative,
    PrimeFieldVectorSpaceBasis,
    _validation_error,
)
from jacobian.math.matrices.finite_fields import PrimeFieldMatrix, rref

MAX_PRIME_FIELD_FLAT_SEARCH_STATE_ORBITS = 100_000
MAX_PRIME_FIELD_FLAT_SEARCH_WORK = 5_000_000_000
MAX_PRIME_FIELD_FLAT_ORBIT_CACHE_ENTRIES = 500_000
_PRIME_FIELD_FLAT_WALL_SECONDS = 3600.0
_CANONICAL_PROJECTION_PASSES = 3

type ResidueRow = tuple[int, ...]
type ClosedCandidateSet = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PrimeFieldFlatPlan:
    deadline: float | None
    candidate_rows: tuple[ResidueRow, ...]
    forbidden_rows: tuple[ResidueRow, ...]
    candidate_permutations: tuple[tuple[int, ...], ...]
    symmetry_group_order: int
    state_orbit_limit: int
    result_orbit_limit: int
    search_work_limit: int
    clause_membership_count: int
    representative_encoding_work: int
    ledger: _WorkLedger


@dataclass(frozen=True, slots=True)
class _FlatState:
    closed_candidates: ClosedCandidateSet
    row_space_basis: tuple[ResidueRow, ...]
    rank: int


class _SearchStoppedError(Exception):
    def __init__(
        self,
        reason: PrimeFieldFlatIncompleteReason,
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
    deadline: float | None
    consumed: int = 0
    state_orbit_count: int = 0
    charged_by_primitive: dict[str, int] = field(default_factory=dict)

    def charge(self, primitive: str, units: int) -> None:
        units = max(units, 1)
        if self.consumed + units > self.limit:
            raise _SearchStoppedError(
                "SEARCH_WORK_LIMIT",
                visited_count=self.state_orbit_count,
                consumed_work=self.consumed,
            )
        self.consumed += units
        self.charged_by_primitive[primitive] = (
            self.charged_by_primitive.get(primitive, 0) + units
        )


def _require_execution_active(deadline: float | None, phase: str) -> None:
    request_checkpoint(phase)
    if deadline is not None and deadline <= time.monotonic():
        raise OperationExecutionTimeoutError(f"request deadline expired {phase}")


def _bind_owner_deadline() -> float:
    execution = current_request_execution()
    started = execution.started_at if execution is not None else time.monotonic()
    owner_deadline = started + _PRIME_FIELD_FLAT_WALL_SECONDS
    deadline = (
        min(owner_deadline, execution.deadline)
        if execution is not None and execution.deadline is not None
        else owner_deadline
    )
    bind_request_deadline(deadline)
    return deadline


def _span_membership_scalar_work(
    *,
    query_count: int,
    rank: int,
    ambient_dimension: int,
) -> int:
    return max(query_count, 1) * max(ambient_dimension, 1) * (2 * max(rank, 1) + 2)


def _subset_container_work(element_count: int, *, sorting: bool = False) -> int:
    size = max(element_count, 1)
    comparison_levels = size.bit_length() if sorting else 1
    return size * (8 * comparison_levels + 8)


def _clause_scan_work(
    problem: ClauseConstrainedPrimeFieldFlatProblem,
    closed_count: int,
    *,
    clause_membership_count: int,
) -> int:
    return (
        max(closed_count, 1)
        + clause_membership_count
        + len(problem.clauses)
        + problem.candidates.vector_count
        + 1
    )


def _modular_row_bits(prime: int) -> int:
    return max(prime.bit_length(), 1)


def _representative_encoding_work(
    problem: ClauseConstrainedPrimeFieldFlatProblem,
) -> int:
    ambient_dimension = len(problem.candidates.coordinate_axis)
    candidate_count = problem.candidates.vector_count
    residue_cells = 2 * ambient_dimension * ambient_dimension
    return (
        4_096
        + candidate_count * (len(str(max(candidate_count, 1))) + 3)
        + residue_cells * (_modular_row_bits(problem.candidates.prime) + 8)
    )


def _admission_work_charges(
    problem: ClauseConstrainedPrimeFieldFlatProblem,
) -> tuple[tuple[str, int], ...]:
    ambient_dimension = len(problem.candidates.coordinate_axis)
    candidate_count = problem.candidates.vector_count
    forbidden_count = problem.forbidden_vectors.row_count
    generator_count = len(problem.symmetry_generators)
    source_cells = (candidate_count + forbidden_count) * ambient_dimension
    residue_bits = _modular_row_bits(problem.candidates.prime)
    normalization_work = 16 * (source_cells + 1) * residue_bits
    compatibility_elements = (
        candidate_count * ambient_dimension
        + MAX_PRIME_FIELD_FLAT_CLAUSE_MEMBERSHIPS
        + forbidden_count * ambient_dimension
        + candidate_count
        + forbidden_count
        + 1
    )
    compatibility_work = generator_count * _subset_container_work(
        compatibility_elements,
        sorting=True,
    )
    paired_degree = ambient_dimension + candidate_count
    group_recognition_work = (
        (generator_count + 3)
        * (MAX_PRIME_FIELD_FLAT_GROUP_ORDER + 1)
        * _subset_container_work(paired_degree)
    )
    source_projection_work = (
        2 * _CANONICAL_PROJECTION_PASSES * compatibility_elements * residue_bits
    )
    return (
        ("admission_normalization", normalization_work + 1),
        ("admission_symmetry", compatibility_work + group_recognition_work),
        ("source_encoding", source_projection_work),
    )


def _permuted_row(
    row: ResidueRow,
    coordinate_permutation: tuple[int, ...],
) -> ResidueRow:
    image = [0] * len(row)
    for source, target in enumerate(coordinate_permutation):
        image[target] = row[source]
    return tuple(image)


def _compose_permutations(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(left[right[index]] for index in range(len(left)))


def _require_symmetry_compatibility(
    problem: ClauseConstrainedPrimeFieldFlatProblem,
) -> tuple[tuple[int, ...], ...]:
    clauses = set(problem.clauses)
    forbidden_rows = set(problem.forbidden_vectors.entries)
    candidate_permutations: list[tuple[int, ...]] = []
    for generator in problem.symmetry_generators:
        coordinate_permutation = tuple(generator.coordinate_permutation)
        candidate_permutation = tuple(generator.candidate_permutation)
        for source, target in enumerate(candidate_permutation):
            if (
                _permuted_row(
                    problem.candidates.vectors.entries[source],
                    coordinate_permutation,
                )
                != problem.candidates.vectors.entries[target]
            ):
                raise _validation_error(
                    "candidate_symmetry",
                    "each paired generator must send every candidate row to its "
                    "declared modular image",
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
            _permuted_row(row, coordinate_permutation) for row in forbidden_rows
        }
        if transformed_forbidden != forbidden_rows:
            raise _validation_error(
                "forbidden_symmetry",
                "each coordinate generator must preserve the forbidden row set",
            )
        candidate_permutations.append(candidate_permutation)
    return tuple(candidate_permutations)


def _paired_group_order(
    problem: ClauseConstrainedPrimeFieldFlatProblem,
    *,
    deadline: float | None,
) -> int:
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
            if len(seen) >= MAX_PRIME_FIELD_FLAT_GROUP_ORDER:
                raise _validation_error(
                    "symmetry_group_order_bound",
                    "the paired symmetry group exceeds the admitted order bound of "
                    f"{MAX_PRIME_FIELD_FLAT_GROUP_ORDER}",
                )
            seen.add(image)
            pending.append(image)
    return len(seen)


def _admit_problem(
    problem: ClauseConstrainedPrimeFieldFlatProblem,
) -> _PrimeFieldFlatPlan:
    deadline = _bind_owner_deadline()
    _require_execution_active(deadline, "before prime-field-flat admission")
    ledger = _WorkLedger(limit=MAX_PRIME_FIELD_FLAT_SEARCH_WORK, deadline=deadline)
    for primitive, units in _admission_work_charges(problem):
        ledger.charge(primitive, units)
    candidate_permutations = _require_symmetry_compatibility(problem)
    _require_execution_active(deadline, "before finite symmetry recognition")
    group_order = _paired_group_order(problem, deadline=deadline)
    _require_execution_active(deadline, "after finite symmetry recognition")
    return _PrimeFieldFlatPlan(
        deadline=deadline,
        candidate_rows=problem.candidates.vectors.entries,
        forbidden_rows=problem.forbidden_vectors.entries,
        candidate_permutations=candidate_permutations,
        symmetry_group_order=group_order,
        state_orbit_limit=MAX_PRIME_FIELD_FLAT_SEARCH_STATE_ORBITS,
        result_orbit_limit=MAX_PRIME_FIELD_FLAT_RESULT_ORBITS,
        search_work_limit=MAX_PRIME_FIELD_FLAT_SEARCH_WORK,
        clause_membership_count=sum(len(clause) for clause in problem.clauses),
        representative_encoding_work=_representative_encoding_work(problem),
        ledger=ledger,
    )


def _rref_basis(
    rows: tuple[ResidueRow, ...],
    *,
    ambient_dimension: int,
    prime: int,
    ledger: _WorkLedger,
) -> tuple[ResidueRow, ...]:
    if not rows:
        return ()
    _require_execution_active(ledger.deadline, "before exact modular row reduction")
    rank_bound = min(len(rows), ambient_dimension)
    input_cells = len(rows) * ambient_dimension
    ledger.charge(
        "row_reduction",
        input_cells * max(rank_bound, 1) + input_cells + rank_bound * ambient_dimension,
    )
    reduced_rows, pivot_columns = rref(
        PrimeFieldMatrix._from_admitted(
            prime=prime, entries=rows, columns=ambient_dimension
        )
    )
    _require_execution_active(ledger.deadline, "after exact modular row reduction")
    # The elimination loop clears each pivot both above and below its pivot
    # row.  Return the final first ``row_index`` rows, not snapshots taken
    # before later pivot columns clear earlier pivot coordinates.
    return reduced_rows[: len(pivot_columns)]


def _pivot_columns(basis: tuple[ResidueRow, ...]) -> tuple[int, ...]:
    return tuple(
        next(index for index, value in enumerate(row) if value) for row in basis
    )


def _row_is_in_span(
    row: ResidueRow,
    basis: tuple[ResidueRow, ...],
    pivots: tuple[int, ...],
    nonpivots: tuple[int, ...],
    prime: int,
) -> bool:
    return all(
        row[column]
        == sum(
            row[pivot] * basis_row[column]
            for basis_row, pivot in zip(basis, pivots, strict=True)
        )
        % prime
        for column in nonpivots
    )


def _closed_candidates(
    candidate_rows: tuple[ResidueRow, ...],
    basis: tuple[ResidueRow, ...],
    *,
    prime: int,
    ledger: _WorkLedger,
) -> ClosedCandidateSet:
    ambient_dimension = (
        len(candidate_rows[0]) if candidate_rows else (len(basis[0]) if basis else 1)
    )
    ledger.charge(
        "candidate_span_scan",
        _span_membership_scalar_work(
            query_count=len(candidate_rows),
            rank=len(basis),
            ambient_dimension=ambient_dimension,
        ),
    )
    pivots = _pivot_columns(basis)
    pivot_set = set(pivots)
    nonpivots = tuple(
        column for column in range(ambient_dimension) if column not in pivot_set
    )
    return tuple(
        index
        for index, row in enumerate(candidate_rows)
        if _row_is_in_span(row, basis, pivots, nonpivots, prime)
    )


class _SubsetOrbitCanonicalizer:
    def __init__(self, generators: tuple[tuple[int, ...], ...]) -> None:
        self._generators = generators
        self._cache: dict[ClosedCandidateSet, tuple[ClosedCandidateSet, int]] = {}

    def canonicalize(
        self, subset: ClosedCandidateSet, ledger: _WorkLedger
    ) -> tuple[ClosedCandidateSet, int]:
        ledger.charge("subset_lookup", _subset_container_work(len(subset)))
        cached = self._cache.get(subset)
        if cached is not None:
            return cached
        if not self._generators:
            ledger.charge("subset_cache", _subset_container_work(len(subset)))
            result = (subset, 1)
            self._cache[subset] = result
            return result

        ledger.charge("subset_storage", 2 * _subset_container_work(len(subset)))
        seen = {subset}
        pending = deque((subset,))
        while pending:
            _require_execution_active(
                ledger.deadline, "during prime-field-flat orbit canonicalization"
            )
            current = pending.popleft()
            for generator in self._generators:
                ledger.charge(
                    "subset_action",
                    _subset_container_work(len(current), sorting=True),
                )
                image = tuple(sorted(generator[index] for index in current))
                if image not in seen:
                    ledger.charge(
                        "subset_storage", 2 * _subset_container_work(len(image))
                    )
                    seen.add(image)
                    pending.append(image)
        retained_elements = len(seen) * max(len(subset), 1)
        ledger.charge(
            "subset_cache", _subset_container_work(retained_elements, sorting=True)
        )
        canonical = min(seen)
        result = (canonical, len(seen))
        if len(self._cache) + len(seen) <= MAX_PRIME_FIELD_FLAT_ORBIT_CACHE_ENTRIES:
            for image in seen:
                self._cache[image] = result
        return result


def _annihilator_basis(
    basis: tuple[ResidueRow, ...], *, ambient_dimension: int, prime: int
) -> tuple[ResidueRow, ...]:
    pivots = _pivot_columns(basis)
    pivot_set = set(pivots)
    vectors: list[ResidueRow] = []
    for free_column in range(ambient_dimension):
        if free_column in pivot_set:
            continue
        vector = [0] * ambient_dimension
        vector[free_column] = 1
        for row_index, pivot in enumerate(pivots):
            vector[pivot] = (-basis[row_index][free_column]) % prime
        vectors.append(tuple(vector))
    return tuple(vectors)


def _basis_value(
    basis: tuple[ResidueRow, ...], *, prime: int, ambient_dimension: int
) -> PrimeFieldVectorSpaceBasis:
    return PrimeFieldVectorSpaceBasis(
        prime=prime,
        ambient_dimension=ambient_dimension,
        vectors=basis,
    )


def _state_from_closed(
    closed: ClosedCandidateSet,
    *,
    plan: _PrimeFieldFlatPlan,
    prime: int,
    ambient_dimension: int,
    ledger: _WorkLedger,
) -> _FlatState:
    ledger.charge(
        "state_construction",
        _subset_container_work(len(closed))
        + max(len(closed), 1) * max(ambient_dimension, 1),
    )
    basis = _rref_basis(
        tuple(plan.candidate_rows[index] for index in closed),
        ambient_dimension=ambient_dimension,
        prime=prime,
        ledger=ledger,
    )
    return _FlatState(closed_candidates=closed, row_space_basis=basis, rank=len(basis))


def _canonical_closure(
    rows: tuple[ResidueRow, ...],
    *,
    plan: _PrimeFieldFlatPlan,
    prime: int,
    ambient_dimension: int,
    ledger: _WorkLedger,
    canonicalizer: _SubsetOrbitCanonicalizer,
) -> ClosedCandidateSet:
    basis = _rref_basis(
        rows,
        ambient_dimension=ambient_dimension,
        prime=prime,
        ledger=ledger,
    )
    closed = _closed_candidates(
        plan.candidate_rows,
        basis,
        prime=prime,
        ledger=ledger,
    )
    canonical, _orbit_size = canonicalizer.canonicalize(closed, ledger)
    return canonical


def _contains_forbidden_row(
    state: _FlatState,
    *,
    plan: _PrimeFieldFlatPlan,
    prime: int,
    ambient_dimension: int,
    ledger: _WorkLedger,
) -> bool:
    ledger.charge(
        "forbidden_span_scan",
        _span_membership_scalar_work(
            query_count=len(plan.forbidden_rows),
            rank=state.rank,
            ambient_dimension=ambient_dimension,
        ),
    )
    pivots = _pivot_columns(state.row_space_basis)
    pivot_set = set(pivots)
    nonpivots = tuple(
        column for column in range(ambient_dimension) if column not in pivot_set
    )
    return any(
        _row_is_in_span(
            row,
            state.row_space_basis,
            pivots,
            nonpivots,
            prime,
        )
        for row in plan.forbidden_rows
    )


def _branch_candidates(
    problem: ClauseConstrainedPrimeFieldFlatProblem,
    closed: ClosedCandidateSet,
    ledger: _WorkLedger,
    *,
    clause_membership_count: int,
) -> tuple[int, ...]:
    ledger.charge(
        "clause_scan",
        _clause_scan_work(
            problem,
            len(closed),
            clause_membership_count=clause_membership_count,
        ),
    )
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
    problem: ClauseConstrainedPrimeFieldFlatProblem,
    closed: ClosedCandidateSet,
    ledger: _WorkLedger,
    *,
    clause_membership_count: int,
) -> bool:
    ledger.charge(
        "clause_scan",
        _clause_scan_work(
            problem,
            len(closed),
            clause_membership_count=clause_membership_count,
        ),
    )
    closed_set = set(closed)
    return all(not closed_set.isdisjoint(clause) for clause in problem.clauses)


def _search_satisfying_states(
    problem: ClauseConstrainedPrimeFieldFlatProblem,
    plan: _PrimeFieldFlatPlan,
) -> tuple[
    dict[ClosedCandidateSet, _FlatState],
    set[ClosedCandidateSet],
    _WorkLedger,
    _SubsetOrbitCanonicalizer,
    int,
]:
    ambient_dimension = len(problem.candidates.coordinate_axis)
    prime = problem.candidates.prime
    ledger = plan.ledger
    canonicalizer = _SubsetOrbitCanonicalizer(plan.candidate_permutations)
    visited: set[ClosedCandidateSet] = set()
    satisfying: dict[ClosedCandidateSet, _FlatState] = {}
    satisfying_elements = 0
    initial = _canonical_closure(
        (),
        plan=plan,
        prime=prime,
        ambient_dimension=ambient_dimension,
        ledger=ledger,
        canonicalizer=canonicalizer,
    )
    ledger.charge("search_frontier", 2 * _subset_container_work(len(initial)))
    pending = [initial]
    queued = {initial}
    while pending:
        _require_execution_active(plan.deadline, "during prime-field-flat search")
        ledger.charge("search_frontier", 3 * _subset_container_work(len(pending[-1])))
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
        ledger.charge("search_frontier", _subset_container_work(len(closed)))
        visited.add(closed)
        ledger.state_orbit_count = len(visited)
        state = _state_from_closed(
            closed,
            plan=plan,
            prime=prime,
            ambient_dimension=ambient_dimension,
            ledger=ledger,
        )
        if state.rank > problem.maximum_rank or _contains_forbidden_row(
            state,
            plan=plan,
            prime=prime,
            ambient_dimension=ambient_dimension,
            ledger=ledger,
        ):
            continue
        clauses_satisfied = _clauses_are_satisfied(
            problem,
            closed,
            ledger,
            clause_membership_count=plan.clause_membership_count,
        )
        if clauses_satisfied and state.rank >= problem.minimum_rank:
            retained_work = _subset_container_work(len(closed))
            ledger.charge("search_frontier", retained_work)
            if closed not in satisfying:
                ledger.charge("search_frontier", retained_work)
                satisfying[closed] = state
                satisfying_elements += max(len(closed), 1)
        if state.rank == problem.maximum_rank:
            continue
        child_keys: set[ClosedCandidateSet] = set()
        child_key_elements = 0
        for index in _branch_candidates(
            problem,
            closed,
            ledger,
            clause_membership_count=plan.clause_membership_count,
        ):
            ledger.charge("search_frontier", ambient_dimension + state.rank + 1)
            child = _canonical_closure(
                (*state.row_space_basis, plan.candidate_rows[index]),
                plan=plan,
                prime=prime,
                ambient_dimension=ambient_dimension,
                ledger=ledger,
                canonicalizer=canonicalizer,
            )
            child_work = _subset_container_work(len(child))
            ledger.charge("search_frontier", child_work)
            if child not in child_keys:
                ledger.charge("search_frontier", child_work)
                child_keys.add(child)
                child_key_elements += max(len(child), 1)
        ledger.charge(
            "search_frontier",
            3 * _subset_container_work(child_key_elements, sorting=True),
        )
        new_children = tuple(
            sorted(
                child
                for child in child_keys
                if child not in visited and child not in queued
            )
        )
        new_child_elements = sum(max(len(child), 1) for child in new_children)
        if len(visited) + len(queued) + len(new_children) > plan.state_orbit_limit:
            raise _SearchStoppedError(
                "STATE_ORBIT_LIMIT",
                visited_count=len(visited),
                consumed_work=ledger.consumed,
            )
        ledger.charge("search_frontier", 2 * _subset_container_work(new_child_elements))
        queued.update(new_children)
        pending.extend(reversed(new_children))
    return satisfying, visited, ledger, canonicalizer, satisfying_elements


def _representatives_from_states(
    satisfying: dict[ClosedCandidateSet, _FlatState],
    *,
    satisfying_elements: int,
    plan: _PrimeFieldFlatPlan,
    prime: int,
    ambient_dimension: int,
    ledger: _WorkLedger,
    canonicalizer: _SubsetOrbitCanonicalizer,
) -> tuple[tuple[PrimeFieldFlatOrbitRepresentative, ...], int]:
    representatives: list[PrimeFieldFlatOrbitRepresentative] = []
    solution_flat_count = 0
    ledger.charge(
        "result_construction",
        _subset_container_work(satisfying_elements, sorting=True),
    )
    for closed, state in sorted(satisfying.items()):
        _require_execution_active(
            ledger.deadline, "during prime-field-flat representative construction"
        )
        _canonical, orbit_size = canonicalizer.canonicalize(closed, ledger)
        ledger.charge(
            "result_construction",
            (
                2 * ambient_dimension * ambient_dimension
                + state.rank * ambient_dimension
                + len(closed)
                + 16
            ),
        )
        representative = PrimeFieldFlatOrbitRepresentative._from_kernel(
            closed_candidate_indices=closed,
            rank=state.rank,
            row_space_basis=_basis_value(
                state.row_space_basis,
                prime=prime,
                ambient_dimension=ambient_dimension,
            ),
            annihilator_basis=_basis_value(
                _annihilator_basis(
                    state.row_space_basis,
                    ambient_dimension=ambient_dimension,
                    prime=prime,
                ),
                prime=prime,
                ambient_dimension=ambient_dimension,
            ),
            orbit_size=orbit_size,
            stabilizer_order=plan.symmetry_group_order // orbit_size,
        )
        ledger.charge(
            "result_encoding",
            2 * _CANONICAL_PROJECTION_PASSES * plan.representative_encoding_work,
        )
        representatives.append(representative)
        solution_flat_count += orbit_size
    return tuple(representatives), solution_flat_count


def _incomplete_result(
    problem: ClauseConstrainedPrimeFieldFlatProblem,
    plan: _PrimeFieldFlatPlan,
    *,
    reason: PrimeFieldFlatIncompleteReason,
    visited_count: int,
    consumed_search_work: int,
) -> ClauseConstrainedPrimeFieldFlatClassification:
    return ClauseConstrainedPrimeFieldFlatClassification._incomplete_from_kernel(
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
    problem: ClauseConstrainedPrimeFieldFlatProblem,
    plan: _PrimeFieldFlatPlan,
) -> ClauseConstrainedPrimeFieldFlatClassification:
    try:
        (
            satisfying,
            _visited,
            ledger,
            canonicalizer,
            satisfying_elements,
        ) = _search_satisfying_states(problem, plan)
        ambient_dimension = len(problem.candidates.coordinate_axis)
        representatives, solution_flat_count = _representatives_from_states(
            satisfying,
            satisfying_elements=satisfying_elements,
            plan=plan,
            prime=problem.candidates.prime,
            ambient_dimension=ambient_dimension,
            ledger=ledger,
            canonicalizer=canonicalizer,
        )
    except _SearchStoppedError as stopped:
        return _incomplete_result(
            problem,
            plan,
            reason=stopped.reason,
            visited_count=stopped.visited_count,
            consumed_search_work=stopped.consumed_work,
        )
    result = ClauseConstrainedPrimeFieldFlatClassification._complete_from_kernel(
        problem=problem,
        symmetry_group_order=plan.symmetry_group_order,
        representatives=representatives,
        solution_flat_count=solution_flat_count,
    )
    _require_execution_active(plan.deadline, "before prime-field-flat result delivery")
    return result


def classify_clause_constrained_prime_field_flats_kernel(
    problem: ClauseConstrainedPrimeFieldFlatProblem,
) -> ClauseConstrainedPrimeFieldFlatClassification:
    """Return complete satisfying prime-field flat orbits when bounded search ends."""

    plan = _admit_problem(problem)
    return _classify(problem, plan)


__all__ = ["classify_clause_constrained_prime_field_flats_kernel"]
