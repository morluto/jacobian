"""Complete clause-constrained rational-flat orbit classification."""

import time
from dataclasses import replace
from fractions import Fraction
from threading import Event
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from sympy import Matrix
from tests.fixtures.accounting import assert_charged_work_parity
from tests.support.rational_flats import seven_coordinate_source_problem

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_cancellation,
    request_execution,
)
from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.matroids.rational_flats import (
    ClauseConstrainedRationalFlatClassification,
    ClauseConstrainedRationalFlatProblem,
    RationalFlatRankInterval,
    RationalFlatSymmetryGenerator,
    RationalVectorConfiguration,
    classify_clause_constrained_rational_flats,
    verify_rational_flat_classification,
    verify_rational_flat_representative,
)
from jacobian.math.combinatorics.matroids.rational_flats import _kernel as flat_kernel
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


def _minimal_raw_request() -> dict[str, Any]:
    return {
        "problem": {
            "candidates": {
                "coordinate_axis": ["x"],
                "vector_labels": ["v"],
                "vectors": {
                    "domain": "QQ",
                    "row_count": 1,
                    "column_count": 1,
                    "entries": [],
                },
            },
            "clauses": [],
            "forbidden_vectors": {
                "domain": "QQ",
                "row_count": 1,
                "column_count": 1,
                "entries": [],
            },
            "rank_interval": None,
            "symmetry_generators": [],
        }
    }


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


def test_complete_family_is_deterministic_and_round_trips() -> None:
    dimension = 8
    rows = tuple(
        tuple(int(row == column) for column in range(dimension))
        for row in range(dimension)
    )
    problem = _problem(rows, columns=dimension)

    first = classify_clause_constrained_rational_flats(problem)
    second = classify_clause_constrained_rational_flats(problem)

    assert first == second
    assert first.outcome.status == "COMPLETE_EXACT"
    assert first.outcome.orbit_count == 2**dimension
    assert first.outcome.solution_flat_count == 2**dimension
    assert (
        ClauseConstrainedRationalFlatClassification.model_validate_json(
            first.model_dump_json()
        )
        == first
    )


def test_serialized_complete_claim_and_representative_are_verifiable() -> None:
    result = classify_clause_constrained_rational_flats(
        _problem(((1,),), columns=1, rank_interval=(1, 1))
    )
    decoded = ClauseConstrainedRationalFlatClassification.model_validate_json(
        result.model_dump_json()
    )
    assert verify_rational_flat_classification(decoded)
    representative = decoded.outcome.representatives[0]
    assert verify_rational_flat_representative(decoded, representative)

    forged = result.model_dump(mode="json")
    forged["outcome"]["representatives"][0]["orbit_size"] = 2
    forged_result = ClauseConstrainedRationalFlatClassification.model_validate(forged)
    assert not verify_rational_flat_classification(forged_result)


def test_request_and_complete_result_round_trip_through_strict_json() -> None:
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


def test_complete_family_requires_distinct_canonical_representative_keys() -> None:
    result = classify_clause_constrained_rational_flats(
        _problem(((1,),), columns=1, rank_interval=(1, 1))
    )

    assert result.outcome.status == "COMPLETE_EXACT"
    representative = result.outcome.representatives[0]
    forged = result.model_dump(mode="json")
    forged["outcome"]["representatives"] = [
        forged["outcome"]["representatives"][0],
        forged["outcome"]["representatives"][0],
    ]
    forged["outcome"]["orbit_count"] = 2
    forged["outcome"]["solution_flat_count"] = 2 * representative.orbit_size
    forged_result = ClauseConstrainedRationalFlatClassification.model_validate(forged)
    assert not verify_rational_flat_classification(forged_result)


@pytest.mark.parametrize("matrix_owner", ["candidate", "forbidden"])
def test_raw_257_digit_components_are_rejected_by_the_outer_input_envelope(
    matrix_owner: str,
) -> None:
    oversized = "9" * 257
    candidate_entries: list[dict[str, object]] = []
    forbidden_entries: list[dict[str, object]] = []
    target = candidate_entries if matrix_owner == "candidate" else forbidden_entries
    target.append(
        {
            "row": 0,
            "column": 0,
            "value": {"num": oversized, "den": "1"},
        }
    )
    raw = _minimal_raw_request()
    raw_problem = raw["problem"]
    raw_problem["candidates"]["vectors"]["entries"] = candidate_entries
    raw_problem["forbidden_vectors"]["entries"] = forbidden_entries

    # The shared rational carrier can represent this value; the operation's
    # narrower error therefore demonstrates rejection in the outer raw pass.
    assert CanonicalRational(num=oversized, den="1").num == oversized
    with pytest.raises(ValidationError) as caught:
        ClauseConstrainedRationalFlatRequest.model_validate(raw)
    assert caught.value.errors()[0]["type"] == "rational_flat.input_component_bound"


def test_raw_sparse_axes_and_nonzeros_are_rejected_before_nested_copying() -> None:
    oversized_rows = _minimal_raw_request()
    oversized_rows["problem"]["candidates"]["vectors"]["row_count"] = 129
    with pytest.raises(ValidationError) as caught_rows:
        ClauseConstrainedRationalFlatRequest.model_validate(oversized_rows)
    assert caught_rows.value.errors()[0]["type"] == "rational_flat.candidate_row_bound"

    oversized_axis = _minimal_raw_request()
    oversized_axis["problem"]["forbidden_vectors"]["column_count"] = 17
    with pytest.raises(ValidationError) as caught_axis:
        ClauseConstrainedRationalFlatRequest.model_validate(oversized_axis)
    assert (
        caught_axis.value.errors()[0]["type"] == "rational_flat.forbidden_column_bound"
    )

    oversized_support = _minimal_raw_request()
    oversized_support["problem"]["candidates"]["vectors"]["entries"] = [
        {"row": 0, "column": 0, "value": {"num": "1", "den": "1"}}
    ] * 2_049
    with pytest.raises(ValidationError) as caught_support:
        ClauseConstrainedRationalFlatRequest.model_validate(oversized_support)
    assert caught_support.value.errors()[0]["type"] == (
        "rational_flat.candidate_nonzero_bound"
    )


@pytest.mark.parametrize(
    ("field_name", "error_type", "oversized"),
    [
        (
            "coordinate_axis",
            "rational_flat.candidate_axis_bound",
            ["x"] * 17,
        ),
        (
            "vector_labels",
            "rational_flat.candidate_label_bound",
            ["v"] * 129,
        ),
    ],
)
def test_raw_candidate_configuration_is_bounded_before_nested_copying(
    field_name: str,
    error_type: str,
    oversized: list[str],
) -> None:
    raw = _minimal_raw_request()
    raw["problem"]["candidates"][field_name] = oversized

    with pytest.raises(ValidationError) as caught:
        ClauseConstrainedRationalFlatRequest.model_validate(raw)

    assert caught.value.errors()[0]["type"] == error_type


@pytest.mark.parametrize("field_name", ["coordinate_axis", "vector_labels"])
def test_raw_configuration_axes_must_be_arrays_before_nested_copying(
    field_name: str,
) -> None:
    raw = _minimal_raw_request()
    raw["problem"]["candidates"][field_name] = {"nested": "container"}

    with pytest.raises(ValidationError) as caught:
        ClauseConstrainedRationalFlatRequest.model_validate(raw)

    assert caught.value.errors()[0]["type"] == "rational_flat.configuration_shape"


def test_raw_repeated_minus_prefix_is_bounded_before_canonicalization() -> None:
    raw = _minimal_raw_request()
    raw["problem"]["forbidden_vectors"]["entries"] = [
        {
            "row": 0,
            "column": 0,
            "value": {"num": "-" * 1_000_000 + "1", "den": "1"},
        }
    ]

    with pytest.raises(ValidationError) as caught:
        ClauseConstrainedRationalFlatRequest.model_validate(raw)

    assert caught.value.errors()[0]["type"] == "rational_flat.input_component_bound"


def test_raw_symmetry_payload_is_bounded_before_nested_copying() -> None:
    oversized_permutation = _minimal_raw_request()
    oversized_permutation["problem"]["symmetry_generators"] = [
        {
            "coordinate_permutation": [0],
            "candidate_permutation": list(range(100_000)),
        }
    ]
    with pytest.raises(ValidationError) as caught_permutation:
        ClauseConstrainedRationalFlatRequest.model_validate(oversized_permutation)
    assert caught_permutation.value.errors()[0]["type"] == (
        "rational_flat.candidate_permutation_bound"
    )

    unknown_nested_field = _minimal_raw_request()
    unknown_nested_field["problem"]["symmetry_generators"] = [
        {
            "coordinate_permutation": [0],
            "candidate_permutation": [0],
            "unknown": [0] * 100_000,
        }
    ]
    with pytest.raises(ValidationError) as caught_unknown:
        ClauseConstrainedRationalFlatRequest.model_validate(unknown_nested_field)
    assert caught_unknown.value.errors()[0]["type"] == (
        "rational_flat.symmetry_generator_shape"
    )


@pytest.mark.parametrize("container_kind", ["recursive", "too_deep"])
def test_raw_recursive_or_excessively_nested_containers_are_typed_rejections(
    container_kind: str,
) -> None:
    raw = _minimal_raw_request()
    clauses: list[Any] = []
    if container_kind == "recursive":
        clauses.append(clauses)
    else:
        nested: list[Any] = []
        for _ in range(CanonicalLimits().max_depth + 1):
            nested = [nested]
        clauses.append(nested)
    raw["problem"]["clauses"] = clauses

    with pytest.raises(ValidationError) as caught:
        ClauseConstrainedRationalFlatRequest.model_validate(raw)

    assert caught.value.errors()[0]["type"] == ("rational_flat.raw_container_structure")


def test_orbit_traversal_charges_exact_units_before_cache_mutation() -> None:
    unit = flat_kernel._subset_container_work(1)
    stopped_ledger = flat_kernel._WorkLedger(
        limit=unit,
        linear_algebra_chunk_cost=1,
        deadline=None,
    )
    stopped_canonicalizer = flat_kernel._SubsetOrbitCanonicalizer(())

    with pytest.raises(flat_kernel._SearchStoppedError) as stopped:
        stopped_canonicalizer.canonicalize((0,), stopped_ledger)

    assert stopped.value.reason == "SEARCH_WORK_LIMIT"
    assert stopped_ledger.consumed == unit
    assert stopped_ledger.charged_by_primitive == {"subset_lookup": unit}
    assert stopped_canonicalizer._cache == {}

    ledger = flat_kernel._WorkLedger(
        limit=10_000,
        linear_algebra_chunk_cost=1,
        deadline=None,
    )
    canonicalizer = flat_kernel._SubsetOrbitCanonicalizer(((1, 0),))

    assert canonicalizer.canonicalize((0,), ledger) == ((0,), 2)
    assert ledger.charged_by_primitive == {
        "subset_lookup": unit,
        "subset_storage": 6 * unit,
        "subset_action": 2 * unit,
        "subset_cache": flat_kernel._subset_container_work(2, sorting=True),
    }
    assert set(canonicalizer._cache) == {(0,), (1,)}


def test_search_work_limit_is_a_typed_stop_at_the_next_exact_charge() -> None:
    problem = _problem((), columns=1)
    plan = flat_kernel._admit_problem(problem)
    admission_work = plan.ledger.consumed
    initial_span_work = (
        flat_kernel._span_membership_scalar_work(
            query_count=0,
            rank=0,
            ambient_dimension=1,
        )
        * plan.linear_algebra_chunk_cost
    )
    subset_unit = flat_kernel._subset_container_work(0)
    exact_cutoff = admission_work + initial_span_work + 2 * subset_unit
    plan.ledger.limit = exact_cutoff
    plan = replace(plan, search_work_limit=exact_cutoff)

    result = flat_kernel._classify(problem, plan)

    assert result.outcome.status == "INCOMPLETE"
    assert result.outcome.reason == "SEARCH_WORK_LIMIT"
    assert result.outcome.explored_state_orbit_count == 0
    assert result.outcome.consumed_search_work == exact_cutoff
    assert result.outcome.search_work_limit == exact_cutoff
    assert not verify_rational_flat_classification(result)


def test_state_orbit_limit_stops_before_retaining_the_next_frontier() -> None:
    problem = _problem(((1,),), columns=1)
    plan = replace(flat_kernel._admit_problem(problem), state_orbit_limit=1)

    result = flat_kernel._classify(problem, plan)

    assert result.outcome.status == "INCOMPLETE"
    assert result.outcome.reason == "STATE_ORBIT_LIMIT"
    assert result.outcome.explored_state_orbit_count == 1
    assert result.outcome.state_orbit_limit == 1
    assert not verify_rational_flat_classification(result)


def test_one_request_ledger_charges_every_observed_work_primitive() -> None:
    problem = _problem(
        ((1, 0), (0, 1), (1, 1)),
        columns=2,
        clauses=((2,),),
        forbidden_rows=((1, -1),),
        rank_interval=(0, 2),
        symmetry_generators=(
            RationalFlatSymmetryGenerator(
                coordinate_permutation=(1, 0),
                candidate_permutation=(1, 0, 2),
            ),
        ),
    )
    executed: dict[str, int] = {}

    def add_work(primitive: str, units: int = 1) -> None:
        executed[primitive] = executed.get(primitive, 0) + units

    original_canonicalize = flat_kernel._SubsetOrbitCanonicalizer.canonicalize
    original_state_from_closed = flat_kernel._state_from_closed

    def counted_canonicalize(
        canonicalizer: flat_kernel._SubsetOrbitCanonicalizer,
        subset: tuple[int, ...],
        ledger: flat_kernel._WorkLedger,
    ) -> tuple[tuple[int, ...], int]:
        was_cached = subset in canonicalizer._cache
        canonical, orbit_size = original_canonicalize(
            canonicalizer,
            subset,
            ledger,
        )
        add_work("subset_lookup")
        if not was_cached:
            add_work("subset_cache")
            if canonicalizer._generators:
                add_work("subset_storage", 3 * orbit_size)
                add_work(
                    "subset_action",
                    orbit_size * len(canonicalizer._generators),
                )
        return canonical, orbit_size

    def counted_state_from_closed(
        closed: tuple[int, ...],
        *,
        plan: flat_kernel._RationalFlatPlan,
        ambient_dimension: int,
        ledger: flat_kernel._WorkLedger,
    ) -> flat_kernel._FlatState:
        add_work("state_construction", 1)
        return original_state_from_closed(
            closed,
            plan=plan,
            ambient_dimension=ambient_dimension,
            ledger=ledger,
        )

    with (
        patch.object(
            flat_kernel,
            "_primitive_integer_row",
            wraps=flat_kernel._primitive_integer_row,
        ) as normalizations,
        patch.object(
            flat_kernel,
            "_require_symmetry_compatibility",
            wraps=flat_kernel._require_symmetry_compatibility,
        ) as compatibility_checks,
        patch.object(
            flat_kernel,
            "_paired_group_order",
            wraps=flat_kernel._paired_group_order,
        ) as group_recognitions,
        patch.object(
            flat_kernel,
            "_rref_basis",
            wraps=flat_kernel._rref_basis,
        ) as row_reductions,
        patch.object(
            flat_kernel,
            "_closed_candidates",
            wraps=flat_kernel._closed_candidates,
        ) as candidate_scans,
        patch.object(
            flat_kernel._SubsetOrbitCanonicalizer,
            "canonicalize",
            new=counted_canonicalize,
        ),
        patch.object(
            flat_kernel,
            "_state_from_closed",
            new=counted_state_from_closed,
        ),
        patch.object(
            flat_kernel,
            "_contains_forbidden_row",
            wraps=flat_kernel._contains_forbidden_row,
        ) as forbidden_scans,
        patch.object(
            flat_kernel,
            "_branch_candidates",
            wraps=flat_kernel._branch_candidates,
        ) as branch_scans,
        patch.object(
            flat_kernel,
            "_clauses_are_satisfied",
            wraps=flat_kernel._clauses_are_satisfied,
        ) as satisfaction_scans,
        patch.object(
            flat_kernel,
            "_canonical_closure",
            wraps=flat_kernel._canonical_closure,
        ) as closures,
    ):
        plan = flat_kernel._admit_problem(problem)
        admission_work = plan.ledger.consumed
        ledger_identity = id(plan.ledger)
        satisfying, visited, ledger, canonicalizer, satisfying_elements = (
            flat_kernel._search_satisfying_states(problem, plan)
        )
        representatives, _ = flat_kernel._representatives_from_states(
            satisfying,
            satisfying_elements=satisfying_elements,
            plan=plan,
            ambient_dimension=2,
            ledger=ledger,
            canonicalizer=canonicalizer,
        )

    executed.update(
        {
            "admission_normalization": normalizations.call_count,
            "admission_symmetry": (
                compatibility_checks.call_count + group_recognitions.call_count
            ),
            "candidate_span_scan": candidate_scans.call_count,
            "clause_scan": branch_scans.call_count + satisfaction_scans.call_count,
            "forbidden_span_scan": forbidden_scans.call_count,
            "result_construction": 1 + len(representatives),
            "row_reduction": sum(
                bool(call.args[0]) for call in row_reductions.call_args_list
            ),
            "search_frontier": (
                len(visited)
                + closures.call_count
                + branch_scans.call_count
                + len(satisfying)
            ),
            "source_encoding": plan.ledger.charged_by_primitive["source_encoding"],
        }
    )
    executed["result_encoding"] = (
        2
        * flat_kernel._CANONICAL_PROJECTION_PASSES
        * len(representatives)
        * plan.representative_encoding_work
    )

    assert representatives
    assert id(ledger) == ledger_identity
    assert (
        ledger.charged_by_primitive["admission_normalization"]
        + ledger.charged_by_primitive["admission_symmetry"]
        + ledger.charged_by_primitive["source_encoding"]
        == admission_work
    )
    assert ledger.consumed == sum(ledger.charged_by_primitive.values())
    assert (
        set(executed)
        == set(ledger.charged_by_primitive)
        == {
            "admission_normalization",
            "admission_symmetry",
            "candidate_span_scan",
            "clause_scan",
            "forbidden_span_scan",
            "result_construction",
            "result_encoding",
            "row_reduction",
            "search_frontier",
            "source_encoding",
            "state_construction",
            "subset_action",
            "subset_cache",
            "subset_lookup",
            "subset_storage",
        }
    )
    assert_charged_work_parity(
        charged=ledger.charged_by_primitive,
        executed=executed,
    )
    assert ledger.charged_by_primitive["result_encoding"] == (
        2
        * flat_kernel._CANONICAL_PROJECTION_PASSES
        * len(representatives)
        * plan.representative_encoding_work
    )


def test_span_work_combines_rref_and_forbidden_row_heights() -> None:
    base = _problem(((1, 0),), columns=2, forbidden_rows=((0, 1),))
    large_forbidden = SparseRationalMatrix(
        row_count=1,
        column_count=2,
        entries=(
            SparseRationalMatrixEntry(
                row=0,
                column=0,
                value=CanonicalRational.from_fraction(Fraction(1, 10**200)),
            ),
            SparseRationalMatrixEntry(
                row=0,
                column=1,
                value=CanonicalRational.from_fraction(Fraction(1, 10**200 - 1)),
            ),
        ),
    )
    large = ClauseConstrainedRationalFlatProblem(
        candidates=base.candidates,
        clauses=base.clauses,
        forbidden_vectors=large_forbidden,
        rank_interval=base.rank_interval,
        symmetry_generators=base.symmetry_generators,
    )

    assert flat_kernel._admit_problem(large).linear_algebra_chunk_cost > (
        flat_kernel._admit_problem(base).linear_algebra_chunk_cost
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


def test_timeout_and_cancellation_are_observed_after_admission() -> None:
    problem = _problem(((1, 0), (0, 1), (1, 1)), columns=2)
    clock_values = iter((*([100.0] * 4), *([106.0] * 20)))
    with (
        request_execution(100.0),
        patch.object(time, "monotonic", side_effect=clock_values),
        pytest.raises(OperationExecutionTimeoutError, match="request deadline expired"),
    ):
        bind_request_deadline(105.0)
        classify_clause_constrained_rational_flats(problem)

    class CancelDuringSearch:
        def __init__(self) -> None:
            self.checks = 0

        def is_set(self) -> bool:
            self.checks += 1
            return self.checks >= 5

    cancellation = CancelDuringSearch()
    with (
        request_cancellation(cancellation),
        pytest.raises(
            OperationExecutionCancelledError,
            match="during rational-flat search",
        ),
    ):
        classify_clause_constrained_rational_flats(problem)


def test_seven_coordinate_regression_has_348_rooted_flats() -> None:
    problem = seven_coordinate_source_problem()

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
    assert sum(rooted_incidence) == 348

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
