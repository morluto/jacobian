"""Exact bounded kernel for clause-constrained rational flats."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from math import factorial, gcd, lcm

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_cancelled,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.math.combinatorics.matroids.rational_flats._models import (
    MAX_RATIONAL_FLAT_CLAUSE_MEMBERSHIPS,
    MAX_RATIONAL_FLAT_GROUP_ORDER,
    MAX_RATIONAL_FLAT_INPUT_COMPONENT_DIGITS,
    MAX_RATIONAL_FLAT_RESULT_BYTES,
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

MAX_RATIONAL_FLAT_SEARCH_STATE_ORBITS = 100_000
MAX_RATIONAL_FLAT_SEARCH_WORK = 5_000_000_000
MAX_RATIONAL_FLAT_ORBIT_CACHE_ENTRIES = 500_000
_RATIONAL_FLAT_WALL_SECONDS = 300.0
_RESULT_ENVELOPE_RESERVE_BYTES = 16_384
_CANONICAL_PROJECTION_PASSES = 3

type RationalRow = tuple[Fraction, ...]
type IntegerRow = tuple[int, ...]
type ClosedCandidateSet = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _RationalFlatPlan:
    """Single-invocation plan owning the request's mutable execution ledger."""

    deadline: float | None
    candidate_rows: tuple[IntegerRow, ...]
    forbidden_rows: tuple[IntegerRow, ...]
    candidate_permutations: tuple[tuple[int, ...], ...]
    symmetry_group_order: int
    state_orbit_limit: int
    result_orbit_limit: int
    result_output_byte_limit: int
    search_work_limit: int
    linear_algebra_chunk_cost: int
    clause_membership_count: int
    source_bytes: int
    representative_encoding_work: int
    ledger: _WorkLedger


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

    def charge_linear_algebra(self, primitive: str, scalar_operations: int) -> None:
        self.charge(
            primitive,
            scalar_operations * self.linear_algebra_chunk_cost,
        )


def _require_execution_active(deadline: float | None, phase: str) -> None:
    if request_cancelled():
        raise OperationExecutionCancelledError(f"request cancelled {phase}")
    if deadline is not None and deadline <= time.monotonic():
        raise OperationExecutionTimeoutError(f"request deadline expired {phase}")


def _bind_owner_deadline() -> float:
    execution = current_request_execution()
    started = execution.started_at if execution is not None else time.monotonic()
    owner_deadline = started + _RATIONAL_FLAT_WALL_SECONDS
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
    """Bound pivot discovery, residual updates, and zero tests for span queries."""

    return max(query_count, 1) * max(ambient_dimension, 1) * (2 * max(rank, 1) + 2)


def _subset_container_work(element_count: int, *, sorting: bool = False) -> int:
    """Bound tuple construction, hashing, set/queue traffic, and optional sorting."""

    size = max(element_count, 1)
    comparison_levels = size.bit_length() if sorting else 1
    return size * (8 * comparison_levels + 8)


def _clause_scan_work(
    problem: ClauseConstrainedRationalFlatProblem,
    closed_count: int,
    *,
    clause_membership_count: int,
) -> int:
    """Bound closed-set construction plus every clause/candidate membership test."""

    return (
        max(closed_count, 1)
        + clause_membership_count
        + len(problem.clauses)
        + problem.candidates.vector_count
        + 1
    )


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


def _admission_work_charges(
    problem: ClauseConstrainedRationalFlatProblem,
) -> tuple[tuple[str, int], ...]:
    """Precharge every bounded admission phase without first scanning source data."""

    ambient_dimension = len(problem.candidates.coordinate_axis)
    candidate_count = problem.candidates.vector_count
    forbidden_count = problem.forbidden_vectors.row_count
    generator_count = len(problem.symmetry_generators)
    source_cells = (candidate_count + forbidden_count) * ambient_dimension
    source_nonzeros = len(problem.candidates.vectors.entries) + len(
        problem.forbidden_vectors.entries
    )
    # Primitive-row normalization can multiply up to one admitted denominator
    # per coordinate.  Charge the resulting multi-precision chunk width before
    # reading components, materializing dense rows, or computing gcds/minors.
    normalization_chunks = (
        ambient_dimension * MAX_RATIONAL_FLAT_INPUT_COMPONENT_DIGITS + 31
    ) // 32
    normalization_work = (
        8
        * (source_cells + source_nonzeros + candidate_count + forbidden_count + 1)
        * max(normalization_chunks * normalization_chunks, 1)
    )
    compatibility_elements = (
        candidate_count * ambient_dimension
        + MAX_RATIONAL_FLAT_CLAUSE_MEMBERSHIPS
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
        * (MAX_RATIONAL_FLAT_GROUP_ORDER + 1)
        * _subset_container_work(paired_degree)
    )
    # Reserve model projection, canonical validation, and RFC encoding both for
    # the admission size check and for final delivery of the retained source.
    source_projection_work = (
        2 * _CANONICAL_PROJECTION_PASSES * CanonicalLimits().max_output_bytes
    )
    return (
        ("admission_normalization", normalization_work + 1),
        (
            "admission_symmetry",
            compatibility_work + group_recognition_work,
        ),
        ("source_encoding", source_projection_work),
    )


def _representative_encoding_work(
    problem: ClauseConstrainedRationalFlatProblem,
    *,
    minor_digits: int,
) -> int:
    """Bound one representative projection for the execution work ledger."""

    ambient_dimension = len(problem.candidates.coordinate_axis)
    candidate_count = problem.candidates.vector_count
    scalar_bytes = 2 * minor_digits + 96
    return (
        4_096
        + candidate_count * (len(str(max(candidate_count, 1))) + 3)
        + 2 * ambient_dimension * ambient_dimension * scalar_bytes
    )


def _complete_result_size(
    *,
    source_bytes: int,
    symmetry_group_order: int,
    representative_count: int,
    representative_bytes: int,
    solution_flat_count: int,
) -> int:
    """Return the exact canonical byte size from accumulated field-value sizes."""

    representatives_size = 2 + max(representative_count - 1, 0) + representative_bytes
    outcome_size = strict_json_object_size(
        (
            ("status", len(encode_strict_json("COMPLETE_EXACT"))),
            ("representatives", representatives_size),
            ("orbit_count", len(encode_strict_json(representative_count))),
            (
                "solution_flat_count",
                len(encode_strict_json(solution_flat_count)),
            ),
        )
    )
    return strict_json_object_size(
        (
            ("problem", source_bytes),
            (
                "symmetry_group_order",
                len(encode_strict_json(symmetry_group_order)),
            ),
            ("outcome", outcome_size),
        )
    )


def _admit_problem(problem: ClauseConstrainedRationalFlatProblem) -> _RationalFlatPlan:
    """Build one pre-search plan for all exact work and result obligations."""

    deadline = _bind_owner_deadline()
    _require_execution_active(deadline, "before rational-flat admission")
    ledger = _WorkLedger(
        limit=MAX_RATIONAL_FLAT_SEARCH_WORK,
        linear_algebra_chunk_cost=1,
        deadline=deadline,
    )
    for primitive, units in _admission_work_charges(problem):
        ledger.charge(primitive, units)
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
    source_row_digits = max(
        _largest_row_component_digits(candidate_rows),
        _largest_row_component_digits(forbidden_rows),
    )
    # Span tests combine an RREF coordinate with a primitive candidate or
    # forbidden coordinate.  Their product height is bounded by the sum, not
    # merely the larger of the two source-derived digit bounds.
    linear_algebra_digits = minor_digits + source_row_digits + 1
    linear_algebra_chunks = (linear_algebra_digits + 31) // 32
    linear_algebra_chunk_cost = linear_algebra_chunks * linear_algebra_chunks
    ledger.linear_algebra_chunk_cost = linear_algebra_chunk_cost
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
    if source_bytes + _RESULT_ENVELOPE_RESERVE_BYTES > (MAX_RATIONAL_FLAT_RESULT_BYTES):
        raise _validation_error(
            "result_size_bound",
            "the retained rational-flat problem leaves no room for a result",
        )
    clause_membership_count = sum(len(clause) for clause in problem.clauses)
    return _RationalFlatPlan(
        deadline=deadline,
        candidate_rows=candidate_rows,
        forbidden_rows=forbidden_rows,
        candidate_permutations=candidate_permutations,
        symmetry_group_order=group_order,
        # The independent state cap bounds retained traversal structure.  The
        # dynamic ledger separately charges every row reduction, span scan,
        # clause scan, subset action, frontier mutation, and result conversion
        # before it executes, so it does not rely on a coarse per-state proxy.
        state_orbit_limit=MAX_RATIONAL_FLAT_SEARCH_STATE_ORBITS,
        result_orbit_limit=MAX_RATIONAL_FLAT_RESULT_ORBITS,
        result_output_byte_limit=MAX_RATIONAL_FLAT_RESULT_BYTES,
        search_work_limit=MAX_RATIONAL_FLAT_SEARCH_WORK,
        linear_algebra_chunk_cost=linear_algebra_chunk_cost,
        clause_membership_count=clause_membership_count,
        source_bytes=source_bytes,
        representative_encoding_work=_representative_encoding_work(
            problem,
            minor_digits=minor_digits,
        ),
        ledger=ledger,
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
    rank_bound = min(len(rows), ambient_dimension)
    input_cells = len(rows) * ambient_dimension
    ledger.charge_linear_algebra(
        "row_reduction",
        input_cells * max(rank_bound, 1) + input_cells + rank_bound * ambient_dimension,
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
        "candidate_span_scan",
        _span_membership_scalar_work(
            query_count=len(candidate_rows),
            rank=len(basis),
            ambient_dimension=ambient_dimension,
        ),
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
                ledger.deadline,
                "during rational-flat orbit canonicalization",
            )
            current = pending[0]
            ledger.charge("subset_storage", _subset_container_work(len(current)))
            current = pending.popleft()
            for generator in self._generators:
                ledger.charge(
                    "subset_action",
                    _subset_container_work(len(current), sorting=True),
                )
                image = tuple(sorted(generator[index] for index in current))
                if image not in seen:
                    ledger.charge(
                        "subset_storage",
                        2 * _subset_container_work(len(image)),
                    )
                    seen.add(image)
                    pending.append(image)
        # Every group image has the source subset's cardinality, so this is the
        # exact retained element count without an uncharged traversal of seen.
        retained_elements = len(seen) * max(len(subset), 1)
        ledger.charge(
            "subset_cache",
            _subset_container_work(retained_elements, sorting=True),
        )
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
) -> _FlatState:
    ledger.charge(
        "state_construction",
        _subset_container_work(len(closed))
        + max(len(closed), 1) * max(ambient_dimension, 1),
    )
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
        "forbidden_span_scan",
        _span_membership_scalar_work(
            query_count=len(plan.forbidden_rows),
            rank=state.rank,
            ambient_dimension=ambient_dimension,
        ),
    )
    return any(
        _row_is_in_span(row, state.row_space_basis) for row in plan.forbidden_rows
    )


def _branch_candidates(
    problem: ClauseConstrainedRationalFlatProblem,
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
    problem: ClauseConstrainedRationalFlatProblem,
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
    problem: ClauseConstrainedRationalFlatProblem,
    plan: _RationalFlatPlan,
) -> tuple[
    dict[ClosedCandidateSet, _FlatState],
    set[ClosedCandidateSet],
    _WorkLedger,
    _SubsetOrbitCanonicalizer,
    int,
]:
    ambient_dimension = len(problem.candidates.coordinate_axis)
    ledger = plan.ledger
    canonicalizer = _SubsetOrbitCanonicalizer(plan.candidate_permutations)
    visited: set[ClosedCandidateSet] = set()
    satisfying: dict[ClosedCandidateSet, _FlatState] = {}
    satisfying_retention_bytes = 0
    satisfying_elements = 0
    initial = _canonical_closure(
        (),
        plan=plan,
        ambient_dimension=ambient_dimension,
        ledger=ledger,
        canonicalizer=canonicalizer,
    )
    ledger.charge("search_frontier", 2 * _subset_container_work(len(initial)))
    pending = [initial]
    queued = {initial}
    while pending:
        _require_execution_active(plan.deadline, "during rational-flat search")
        closed = pending[-1]
        ledger.charge(
            "search_frontier",
            3 * _subset_container_work(len(closed)),
        )
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
            ambient_dimension=ambient_dimension,
            ledger=ledger,
        )
        if state.rank > problem.maximum_rank or _contains_forbidden_row(
            state,
            plan=plan,
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
                # Each retained coordinate is a separately allocated Fraction
                # (object header + numerator pointer + denominator pointer),
                # plus tuple pointer overhead.  Use a conservative 16-byte
                # per-rational bound instead of 8 bytes to account for the
                # Fraction object and container overhead.
                _state_bytes = ambient_dimension * max(state.rank, 1) * 16
                if (
                    satisfying_retention_bytes + _state_bytes
                    > CanonicalLimits().max_output_bytes
                ):
                    raise _SearchStoppedError(
                        "RESULT_OUTPUT_LIMIT",
                        visited_count=ledger.state_orbit_count,
                        consumed_work=ledger.consumed,
                    )
                ledger.charge("search_frontier", retained_work)
                satisfying[closed] = state
                satisfying_elements += max(len(closed), 1)
                satisfying_retention_bytes += _state_bytes
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
        # Iterate only child_keys for the two membership filters and sort.  The
        # retained visited/queued sets are hash lookup targets, not rescanned.
        # This charge also covers the subsequent exact new-child element sum.
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
        ledger.charge(
            "search_frontier",
            2 * _subset_container_work(new_child_elements),
        )
        queued.update(new_children)
        pending.extend(reversed(new_children))
    return satisfying, visited, ledger, canonicalizer, satisfying_elements


def _representatives_from_states(
    satisfying: dict[ClosedCandidateSet, _FlatState],
    *,
    satisfying_elements: int,
    plan: _RationalFlatPlan,
    ambient_dimension: int,
    ledger: _WorkLedger,
    canonicalizer: _SubsetOrbitCanonicalizer,
) -> tuple[tuple[RationalFlatOrbitRepresentative, ...], int]:
    representatives: list[RationalFlatOrbitRepresentative] = []
    representative_bytes = 0
    solution_flat_count = 0
    ledger.charge(
        "result_construction",
        _subset_container_work(satisfying_elements, sorting=True),
    )
    for closed, state in sorted(satisfying.items()):
        _require_execution_active(
            ledger.deadline,
            "during rational-flat representative construction",
        )
        _canonical, orbit_size = canonicalizer.canonicalize(closed, ledger)
        # Two exact bases are converted to canonical rationals.  The
        # annihilator construction additionally reads at most rank*ambient
        # RREF coordinates; all use the admitted minor-height chunk factor.
        ledger.charge_linear_algebra(
            "result_construction",
            (
                2 * ambient_dimension * ambient_dimension
                + state.rank * ambient_dimension
                + len(closed)
                + 16
            ),
        )
        representative = RationalFlatOrbitRepresentative._from_kernel(
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
        # Reserve model projection, canonical validation, and RFC encoding both
        # for this exact-size measurement and for final dispatch delivery.
        ledger.charge(
            "result_encoding",
            2 * _CANONICAL_PROJECTION_PASSES * plan.representative_encoding_work,
        )
        try:
            encoded_representative_size = len(
                encode_strict_json(representative.model_dump(mode="json"))
            )
        except CanonicalizationError:
            raise _SearchStoppedError(
                "RESULT_OUTPUT_LIMIT",
                visited_count=ledger.state_orbit_count,
                consumed_work=ledger.consumed,
            ) from None
        projected_count = len(representatives) + 1
        projected_representative_bytes = (
            representative_bytes + encoded_representative_size
        )
        projected_solution_count = solution_flat_count + orbit_size
        if (
            _complete_result_size(
                source_bytes=plan.source_bytes,
                symmetry_group_order=plan.symmetry_group_order,
                representative_count=projected_count,
                representative_bytes=projected_representative_bytes,
                solution_flat_count=projected_solution_count,
            )
            > plan.result_output_byte_limit
        ):
            raise _SearchStoppedError(
                "RESULT_OUTPUT_LIMIT",
                visited_count=ledger.state_orbit_count,
                consumed_work=ledger.consumed,
            )
        representatives.append(representative)
        representative_bytes = projected_representative_bytes
        solution_flat_count = projected_solution_count
    return tuple(representatives), solution_flat_count


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
        result_output_byte_limit=plan.result_output_byte_limit,
        consumed_search_work=consumed_search_work,
        search_work_limit=plan.search_work_limit,
    )


def _classify(
    problem: ClauseConstrainedRationalFlatProblem,
    plan: _RationalFlatPlan,
) -> ClauseConstrainedRationalFlatClassification:
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
        solution_flat_count=solution_flat_count,
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
