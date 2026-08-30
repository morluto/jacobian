"""Complete clause-constrained rational-flat orbit classification."""

import time
from fractions import Fraction
from itertools import combinations
from threading import Event

import pytest
from sympy import Matrix

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_cancellation,
    request_execution,
)
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.matroids.rational_flats import (
    ClauseConstrainedRationalFlatClassification,
    ClauseConstrainedRationalFlatProblem,
    RationalFlatRankInterval,
    RationalFlatSymmetryGenerator,
    RationalVectorConfiguration,
    classify_clause_constrained_rational_flats,
)
from jacobian.math.combinatorics.matroids.rational_flats._models import (
    ClauseConstrainedRationalFlatRequest,
)
from jacobian.math.matrices.values import (
    SparseRationalMatrix,
    SparseRationalMatrixEntry,
)


def _rational(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _sparse(rows: tuple[tuple[int, ...], ...], *, columns: int) -> SparseRationalMatrix:
    assert all(len(row) == columns for row in rows)
    return SparseRationalMatrix(
        row_count=len(rows),
        column_count=columns,
        entries=tuple(
            SparseRationalMatrixEntry(
                row=row_index,
                column=column_index,
                value=_rational(value),
            )
            for row_index, row in enumerate(rows)
            for column_index, value in enumerate(row)
            if value
        ),
    )


def _dense_rows(matrix: SparseRationalMatrix) -> tuple[tuple[Fraction, ...], ...]:
    rows = [
        [Fraction(0) for _ in range(matrix.column_count)]
        for _ in range(matrix.row_count)
    ]
    for entry in matrix.entries:
        rows[entry.row][entry.column] = entry.value.as_fraction()
    return tuple(tuple(row) for row in rows)


def _problem(
    rows: tuple[tuple[int, ...], ...],
    *,
    columns: int,
    clauses: tuple[tuple[int, ...], ...] = (),
    forbidden_rows: tuple[tuple[int, ...], ...] = (),
    rank_interval: tuple[int, int] | None = None,
    symmetry_generators: tuple[RationalFlatSymmetryGenerator, ...] = (),
) -> ClauseConstrainedRationalFlatProblem:
    return ClauseConstrainedRationalFlatProblem(
        candidates=RationalVectorConfiguration(
            coordinate_axis=tuple(f"x{index}" for index in range(columns)),
            vector_labels=tuple(f"v{index}" for index in range(len(rows))),
            vectors=_sparse(rows, columns=columns),
        ),
        clauses=clauses,
        forbidden_vectors=_sparse(forbidden_rows, columns=columns),
        rank_interval=(
            RationalFlatRankInterval(
                minimum=rank_interval[0],
                maximum=rank_interval[1],
            )
            if rank_interval is not None
            else None
        ),
        symmetry_generators=symmetry_generators,
    )


def _matrix_rank(
    rows: tuple[tuple[int | Fraction, ...], ...],
    columns: int,
) -> int:
    if not rows:
        return 0
    return int(Matrix(rows).rank())


def _brute_force_closed_sets(
    rows: tuple[tuple[int, ...], ...],
    *,
    columns: int,
    clauses: tuple[tuple[int, ...], ...],
    forbidden_rows: tuple[tuple[int, ...], ...],
    minimum_rank: int,
    maximum_rank: int,
) -> tuple[tuple[int, ...], ...]:
    """Independent tiny oracle using SymPy rank rather than the FLINT kernel."""

    closed_sets: set[tuple[int, ...]] = set()
    for mask in range(1 << len(rows)):
        selected = tuple(rows[index] for index in range(len(rows)) if mask >> index & 1)
        selected_rank = _matrix_rank(selected, columns)
        closed = tuple(
            index
            for index, row in enumerate(rows)
            if _matrix_rank((*selected, row), columns) == selected_rank
        )
        closed_sets.add(closed)

    satisfying: list[tuple[int, ...]] = []
    for closed in closed_sets:
        spanning_rows = tuple(rows[index] for index in closed)
        rank = _matrix_rank(spanning_rows, columns)
        if not minimum_rank <= rank <= maximum_rank:
            continue
        if any(set(closed).isdisjoint(clause) for clause in clauses):
            continue
        if any(
            _matrix_rank((*spanning_rows, forbidden), columns) == rank
            for forbidden in forbidden_rows
        ):
            continue
        satisfying.append(closed)
    return tuple(sorted(satisfying))


def _primitive_line(row: tuple[int, ...]) -> tuple[int, ...]:
    first_nonzero = next((value for value in row if value), 0)
    return tuple(-value for value in row) if first_nonzero < 0 else row


def _permuted_row(
    row: tuple[int, ...], permutation: tuple[int, ...]
) -> tuple[int, ...]:
    image = [0] * len(row)
    for source, target in enumerate(permutation):
        image[target] = row[source]
    return _primitive_line(tuple(image))


def _source_problem() -> ClauseConstrainedRationalFlatProblem:
    ambient_dimension = 7
    candidate_rows: list[tuple[int, ...]] = []
    candidate_supports: list[frozenset[int]] = []
    for support in combinations(range(ambient_dimension), 4):
        first, second, third, fourth = support
        for positive, negative in (
            ((first, second), (third, fourth)),
            ((first, third), (second, fourth)),
            ((first, fourth), (second, third)),
        ):
            row = [0] * ambient_dimension
            for coordinate in positive:
                row[coordinate] = 1
            for coordinate in negative:
                row[coordinate] = -1
            candidate_rows.append(_primitive_line(tuple(row)))
            candidate_supports.append(frozenset(support))

    candidates = tuple(candidate_rows)
    candidate_index = {row: index for index, row in enumerate(candidates)}
    assert len(candidates) == len(candidate_index) == 105
    clauses = tuple(
        tuple(
            index
            for index, support in enumerate(candidate_supports)
            if support <= frozenset(five_set)
        )
        for five_set in combinations(range(ambient_dimension), 5)
    )
    assert len(clauses) == 21
    assert {len(clause) for clause in clauses} == {15}

    forbidden_rows = tuple(
        tuple(
            1 if coordinate == first else -1 if coordinate == second else 0
            for coordinate in range(ambient_dimension)
        )
        for first, second in combinations(range(ambient_dimension), 2)
    )
    coordinate_generators = (
        (1, 0, 2, 3, 4, 5, 6),
        (1, 2, 3, 4, 5, 6, 0),
    )
    symmetry_generators = tuple(
        RationalFlatSymmetryGenerator(
            coordinate_permutation=permutation,
            candidate_permutation=tuple(
                candidate_index[_permuted_row(row, permutation)] for row in candidates
            ),
        )
        for permutation in coordinate_generators
    )
    return ClauseConstrainedRationalFlatProblem(
        candidates=RationalVectorConfiguration(
            coordinate_axis=tuple(
                f"a{index + 1}" for index in range(ambient_dimension)
            ),
            vector_labels=tuple(
                f"equation_{index}" for index in range(len(candidates))
            ),
            vectors=_sparse(candidates, columns=ambient_dimension),
        ),
        clauses=clauses,
        forbidden_vectors=_sparse(forbidden_rows, columns=ambient_dimension),
        rank_interval=RationalFlatRankInterval(minimum=4, maximum=5),
        symmetry_generators=symmetry_generators,
    )


def test_satisfied_clause_does_not_stop_satisfying_superflat_enumeration() -> None:
    problem = ClauseConstrainedRationalFlatProblem(
        candidates=RationalVectorConfiguration(
            coordinate_axis=("x", "y"),
            vector_labels=("x_zero", "y_zero", "x_plus_y_zero"),
            vectors=_sparse(((1, 0), (0, 1), (1, 1)), columns=2),
        ),
        clauses=((2,),),
        forbidden_vectors=_sparse((), columns=2),
        rank_interval=RationalFlatRankInterval(minimum=0, maximum=2),
        symmetry_generators=(),
    )

    result = classify_clause_constrained_rational_flats(problem)

    assert result.outcome.status == "COMPLETE_EXACT"
    assert result.symmetry_group_order == 1
    assert result.outcome.solution_flat_count == 2
    assert tuple(
        (representative.closed_candidate_indices, representative.rank)
        for representative in sorted(
            result.outcome.representatives, key=lambda item: item.rank
        )
    ) == (((2,), 1), ((0, 1, 2), 2))


def test_zero_and_parallel_candidates_remain_labelled_matroid_elements() -> None:
    problem = _problem(
        ((0, 0), (1, 0), (2, 0), (0, 1)),
        columns=2,
    )

    result = classify_clause_constrained_rational_flats(problem)

    assert result.outcome.status == "COMPLETE_EXACT"
    assert tuple(
        representative.closed_candidate_indices
        for representative in result.outcome.representatives
    ) == ((0,), (0, 1, 2), (0, 1, 2, 3), (0, 3))
    zero_flat = result.outcome.representatives[0]
    assert zero_flat.rank == 0
    assert zero_flat.row_space_basis.ambient_dimension == 2
    assert zero_flat.row_space_basis.vectors == ()
    assert len(zero_flat.annihilator_basis.vectors) == 2


def test_empty_configuration_has_one_zero_rank_flat() -> None:
    problem = _problem((), columns=2)

    result = classify_clause_constrained_rational_flats(problem)

    assert result.outcome.status == "COMPLETE_EXACT"
    assert result.outcome.solution_flat_count == 1
    representative = result.outcome.representatives[0]
    assert representative.closed_candidate_indices == ()
    assert representative.rank == 0
    assert representative.row_space_basis.vectors == ()
    assert tuple(
        tuple(value.as_fraction() for value in vector)
        for vector in representative.annihilator_basis.vectors
    ) == ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))


@pytest.mark.parametrize(
    ("clauses", "forbidden_rows"),
    [
        (((),), ()),
        ((), ((0, 0),)),
    ],
)
def test_impossible_constraint_returns_an_exact_empty_family(
    clauses: tuple[tuple[int, ...], ...],
    forbidden_rows: tuple[tuple[int, ...], ...],
) -> None:
    problem = _problem(
        ((1, 0),),
        columns=2,
        clauses=clauses,
        forbidden_rows=forbidden_rows,
    )

    result = classify_clause_constrained_rational_flats(problem)

    assert result.outcome.status == "COMPLETE_EXACT"
    assert result.outcome.representatives == ()
    assert result.outcome.orbit_count == result.outcome.solution_flat_count == 0


def test_coordinate_swap_identifies_the_two_coordinate_lines() -> None:
    trivial_problem = _problem(
        ((1, 0), (0, 1)),
        columns=2,
        rank_interval=(1, 1),
    )
    problem = _problem(
        ((1, 0), (0, 1)),
        columns=2,
        rank_interval=(1, 1),
        symmetry_generators=(
            RationalFlatSymmetryGenerator(
                coordinate_permutation=(1, 0),
                candidate_permutation=(1, 0),
            ),
        ),
    )

    trivial_result = classify_clause_constrained_rational_flats(trivial_problem)
    result = classify_clause_constrained_rational_flats(problem)

    assert trivial_result.outcome.status == "COMPLETE_EXACT"
    assert trivial_result.outcome.solution_flat_count == 2
    assert len(trivial_result.outcome.representatives) == 2
    assert result.outcome.status == "COMPLETE_EXACT"
    assert result.symmetry_group_order == 2
    assert result.outcome.solution_flat_count == 2
    assert len(result.outcome.representatives) == 1
    representative = result.outcome.representatives[0]
    assert representative.closed_candidate_indices == (0,)
    assert representative.orbit_size == 2
    assert representative.stabilizer_order == 1


def test_exact_rational_rank_avoids_modular_rank_drop() -> None:
    prime = 1_000_003
    problem = _problem(
        ((1, 0), (1, prime)),
        columns=2,
        clauses=((1,),),
        rank_interval=(1, 2),
    )

    result = classify_clause_constrained_rational_flats(problem)

    assert result.outcome.status == "COMPLETE_EXACT"
    assert tuple(
        (representative.closed_candidate_indices, representative.rank)
        for representative in result.outcome.representatives
    ) == (((0, 1), 2), ((1,), 1))


def test_candidate_action_kernel_is_retained_in_the_stabilizer() -> None:
    problem = _problem(
        ((1, 1),),
        columns=2,
        clauses=((0,),),
        rank_interval=(1, 1),
        symmetry_generators=(
            RationalFlatSymmetryGenerator(
                coordinate_permutation=(1, 0),
                candidate_permutation=(0,),
            ),
        ),
    )

    result = classify_clause_constrained_rational_flats(problem)

    assert result.outcome.status == "COMPLETE_EXACT"
    assert result.symmetry_group_order == 2
    representative = result.outcome.representatives[0]
    assert representative.orbit_size == 1
    assert representative.stabilizer_order == 2


def test_incompatible_paired_symmetry_is_a_typed_domain_error() -> None:
    problem = _problem(
        ((1, 0), (0, 1)),
        columns=2,
        symmetry_generators=(
            RationalFlatSymmetryGenerator(
                coordinate_permutation=(1, 0),
                candidate_permutation=(0, 1),
            ),
        ),
    )

    with pytest.raises(OperationDomainValidationError) as error:
        classify_clause_constrained_rational_flats(problem)

    assert error.value.errors() == (
        {
            "loc": ("problem",),
            "type": "rational_flat.candidate_symmetry",
            "msg": (
                "each paired generator must send every candidate row to its "
                "declared projective image"
            ),
        },
    )


def test_oversized_source_component_is_rejected_before_linear_algebra() -> None:
    problem = _problem(((int("1" * 257),),), columns=1)

    with pytest.raises(OperationDomainValidationError) as error:
        classify_clause_constrained_rational_flats(problem)

    assert error.value.errors()[0]["type"] == "rational_flat.input_component_bound"


def test_oversized_symmetry_group_stops_at_the_admitted_order_boundary() -> None:
    dimension = 8
    identity_candidate = (0,)
    problem = _problem(
        (tuple(0 for _ in range(dimension)),),
        columns=dimension,
        symmetry_generators=(
            RationalFlatSymmetryGenerator(
                coordinate_permutation=(1, 0, *range(2, dimension)),
                candidate_permutation=identity_candidate,
            ),
            RationalFlatSymmetryGenerator(
                coordinate_permutation=(*range(1, dimension), 0),
                candidate_permutation=identity_candidate,
            ),
        ),
    )

    with pytest.raises(OperationDomainValidationError) as error:
        classify_clause_constrained_rational_flats(problem)

    assert error.value.errors()[0]["type"] == (
        "rational_flat.symmetry_group_order_bound"
    )


def test_tiny_search_matches_independent_symbolic_flat_oracle() -> None:
    rows = (
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
    )
    clauses = ((1, 3), (4, 5))
    forbidden_rows = ((1, -1, 0),)
    problem = _problem(
        rows,
        columns=3,
        clauses=clauses,
        forbidden_rows=forbidden_rows,
        rank_interval=(1, 2),
    )

    result = classify_clause_constrained_rational_flats(problem)

    assert result.outcome.status == "COMPLETE_EXACT"
    expected = _brute_force_closed_sets(
        rows,
        columns=3,
        clauses=clauses,
        forbidden_rows=forbidden_rows,
        minimum_rank=1,
        maximum_rank=2,
    )
    assert (
        tuple(
            representative.closed_candidate_indices
            for representative in result.outcome.representatives
        )
        == expected
    )
    assert result.outcome.solution_flat_count == len(expected)


def test_result_orbit_envelope_returns_no_partial_mathematical_family() -> None:
    dimension = 16
    rows = tuple(
        tuple(int(row == column) for column in range(dimension))
        for row in range(dimension)
    )
    problem = _problem(rows, columns=dimension)

    first = classify_clause_constrained_rational_flats(problem)
    second = classify_clause_constrained_rational_flats(problem)

    assert first == second
    assert first.outcome.status == "INCOMPLETE"
    assert first.outcome.reason == "RESULT_ORBIT_LIMIT"
    assert first.outcome.explored_state_orbit_count > 0
    assert first.outcome.result_orbit_limit > 0
    assert first.outcome.consumed_search_work > 0
    assert (
        ClauseConstrainedRationalFlatClassification.model_validate_json(
            first.model_dump_json()
        )
        == first
    )


def test_large_complete_family_fits_the_canonical_transport_envelope() -> None:
    dimension = 8
    rows = tuple(
        tuple(int(row == column) for column in range(dimension))
        for row in range(dimension)
    )

    result = classify_clause_constrained_rational_flats(
        _problem(rows, columns=dimension)
    )

    assert result.outcome.status == "COMPLETE_EXACT"
    assert result.outcome.orbit_count == 2**dimension
    assert len(encode_strict_json(result.model_dump(mode="json"))) <= (
        CanonicalLimits().max_output_bytes
    )


def test_request_and_complete_result_replay_through_strict_json() -> None:
    problem = _problem(
        ((1, 0), (0, 1), (1, 1)),
        columns=2,
        clauses=((2,),),
    )
    request = ClauseConstrainedRationalFlatRequest(problem=problem)
    result = classify_clause_constrained_rational_flats(problem)

    assert (
        ClauseConstrainedRationalFlatRequest.model_validate_json(
            request.model_dump_json()
        )
        == request
    )
    assert (
        ClauseConstrainedRationalFlatClassification.model_validate_json(
            result.model_dump_json()
        )
        == result
    )
    assert ClauseConstrainedRationalFlatRequest.model_json_schema()["required"] == [
        "problem"
    ]
    result_schema = ClauseConstrainedRationalFlatClassification.model_json_schema()
    assert (
        "status"
        in result_schema["$defs"]["RationalFlatClassificationComplete"]["required"]
    )
    assert (
        "status"
        in result_schema["$defs"]["RationalFlatClassificationIncomplete"]["required"]
    )


def test_expired_deadline_and_cancellation_remain_execution_failures() -> None:
    problem = _problem(((1, 0),), columns=2)
    with request_execution(time.monotonic()):
        bind_request_deadline(time.monotonic() - 1)
        with pytest.raises(OperationExecutionTimeoutError, match="admission"):
            classify_clause_constrained_rational_flats(problem)

    cancelled = Event()
    cancelled.set()
    with (
        request_cancellation(cancelled),
        pytest.raises(OperationExecutionCancelledError, match="admission"),
    ):
        classify_clause_constrained_rational_flats(problem)


def test_seven_coordinate_regression_has_two_complete_orbits() -> None:
    problem = _source_problem()

    result = classify_clause_constrained_rational_flats(problem)

    assert result.outcome.status == "COMPLETE_EXACT"
    assert result.symmetry_group_order == 5_040
    assert result.outcome.solution_flat_count == 2_940
    profiles = tuple(
        sorted(
            (
                representative.rank,
                representative.orbit_size,
                representative.stabilizer_order,
                len(representative.closed_candidate_indices),
            )
            for representative in result.outcome.representatives
        )
    )
    assert profiles == ((4, 420, 12, 9), (5, 2_520, 2, 13))
    assert tuple(
        representative.closed_candidate_indices
        for representative in result.outcome.representatives
    ) == (
        (0, 4, 8, 14, 36, 44, 56, 58, 72, 76, 82, 89, 91),
        (0, 4, 16, 43, 55, 59, 77, 83, 92),
    )

    rooted_incidence = tuple(
        representative.orbit_size
        * len(representative.closed_candidate_indices)
        // problem.candidates.vector_count
        for representative in sorted(
            result.outcome.representatives, key=lambda item: item.rank
        )
    )
    assert rooted_incidence == (36, 312)

    candidate_rows = _dense_rows(problem.candidates.vectors)
    forbidden_rows = _dense_rows(problem.forbidden_vectors)
    for representative in result.outcome.representatives:
        assert len(representative.row_space_basis.vectors) == representative.rank
        assert len(representative.annihilator_basis.vectors) == (
            len(problem.candidates.coordinate_axis) - representative.rank
        )
        row_space = tuple(
            tuple(value.as_fraction() for value in row)
            for row in representative.row_space_basis.vectors
        )
        assert _matrix_rank(row_space, 7) == representative.rank
        closed_indices = set(representative.closed_candidate_indices)
        for index, candidate in enumerate(candidate_rows):
            assert (
                _matrix_rank((*row_space, candidate), 7) == representative.rank
            ) == (index in closed_indices)
        assert all(
            _matrix_rank((*row_space, forbidden), 7) > representative.rank
            for forbidden in forbidden_rows
        )
        for row in representative.row_space_basis.vectors:
            for annihilator in representative.annihilator_basis.vectors:
                assert (
                    sum(
                        (
                            left.as_fraction() * right.as_fraction()
                            for left, right in zip(row, annihilator, strict=True)
                        ),
                        Fraction(0),
                    )
                    == 0
                )
