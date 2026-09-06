"""Defining tests for clause-constrained prime-field flat classification."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import cast

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import MathTool
from jacobian.math.combinatorics.matroids.prime_field_flats import (
    classify_clause_constrained_prime_field_flats,
    verify_prime_field_flat_classification,
    verify_prime_field_flat_representative,
)
from jacobian.math.combinatorics.matroids.prime_field_flats._kernel import (
    _admit_problem,
    _classify,
    _subset_container_work,
)
from jacobian.math.combinatorics.matroids.prime_field_flats._models import (
    ClauseConstrainedPrimeFieldFlatClassification,
    ClauseConstrainedPrimeFieldFlatProblem,
    ClauseConstrainedPrimeFieldFlatRequest,
    PrimeFieldFlatRankInterval,
    PrimeFieldFlatSymmetryGenerator,
    PrimeFieldVectorConfiguration,
)
from jacobian.math.combinatorics.matroids.prime_field_flats._tools import TOOLS
from jacobian.math.combinatorics.matroids.rational_flats import (
    RationalFlatRankInterval,
    RationalVectorConfiguration,
    classify_clause_constrained_rational_flats,
)
from jacobian.math.combinatorics.matroids.rational_flats._models import (
    ClauseConstrainedRationalFlatProblem,
)
from jacobian.math.matrices.values import (
    SparseRationalMatrix,
    SparseRationalMatrixEntry,
)


def _operation() -> MathTool[
    ClauseConstrainedPrimeFieldFlatRequest,
    ClauseConstrainedPrimeFieldFlatClassification,
]:
    return cast(
        MathTool[
            ClauseConstrainedPrimeFieldFlatRequest,
            ClauseConstrainedPrimeFieldFlatClassification,
        ],
        next(
            operation
            for operation in TOOLS
            if operation.operation_id
            == "matroid.prime_field_flat.constrained_orbits.compute"
        ),
    )


def _problem(
    rows: tuple[tuple[int, ...], ...],
    *,
    prime: int,
    clauses: tuple[tuple[int, ...], ...] = (),
    forbidden_rows: tuple[tuple[int, ...], ...] = (),
    rank_interval: tuple[int, int] | None = None,
    symmetry_generators: tuple[PrimeFieldFlatSymmetryGenerator, ...] = (),
) -> ClauseConstrainedPrimeFieldFlatProblem:
    columns = len(rows[0]) if rows else 1
    return ClauseConstrainedPrimeFieldFlatProblem(
        candidates=PrimeFieldVectorConfiguration(
            prime=prime,
            coordinate_axis=tuple(f"x{index}" for index in range(columns)),
            vector_labels=tuple(f"v{index}" for index in range(len(rows))),
            vectors={
                "prime": prime,
                "entries": [[value % prime for value in row] for row in rows],
                "columns": columns,
            },  # type: ignore[arg-type]
        ),
        clauses=clauses,
        forbidden_vectors={
            "prime": prime,
            "entries": [[value % prime for value in row] for row in forbidden_rows],
            "columns": columns,
        },  # type: ignore[arg-type]
        rank_interval=(
            PrimeFieldFlatRankInterval(
                minimum=rank_interval[0], maximum=rank_interval[1]
            )
            if rank_interval is not None
            else None
        ),
        symmetry_generators=symmetry_generators,
    )


def _run(
    rows: tuple[tuple[int, ...], ...], *, prime: int, **kwargs: object
) -> ClauseConstrainedPrimeFieldFlatClassification:
    if "clauses" in kwargs:
        kwargs["clauses"] = tuple(kwargs["clauses"])  # type: ignore[arg-type]
    if "forbidden_rows" in kwargs:
        kwargs["forbidden_rows"] = tuple(kwargs["forbidden_rows"])  # type: ignore[arg-type]
    if "symmetry_generators" in kwargs:
        kwargs["symmetry_generators"] = tuple(kwargs["symmetry_generators"])  # type: ignore[arg-type]
    problem = _problem(rows, prime=prime, **kwargs)  # type: ignore[arg-type]
    return classify_clause_constrained_prime_field_flats(problem)


def _sparse_rational(
    rows: tuple[tuple[int, ...], ...], *, columns: int
) -> SparseRationalMatrix:
    return SparseRationalMatrix(
        row_count=len(rows),
        column_count=columns,
        entries=tuple(
            SparseRationalMatrixEntry(
                row=row,
                column=column,
                value=CanonicalRational.from_fraction(Fraction(value)),
            )
            for row, values in enumerate(rows)
            for column, value in enumerate(values)
            if value
        ),
    )


def _rational_problem(
    rows: tuple[tuple[int, ...], ...],
    *,
    clauses: tuple[tuple[int, ...], ...] = (),
    forbidden_rows: tuple[tuple[int, ...], ...] = (),
    rank_interval: tuple[int, int] | None = None,
) -> ClauseConstrainedRationalFlatProblem:
    columns = len(rows[0])
    return ClauseConstrainedRationalFlatProblem(
        candidates=RationalVectorConfiguration(
            coordinate_axis=tuple(f"x{index}" for index in range(columns)),
            vector_labels=tuple(f"v{index}" for index in range(len(rows))),
            vectors=_sparse_rational(rows, columns=columns),
        ),
        clauses=clauses,
        forbidden_vectors=_sparse_rational(forbidden_rows, columns=columns),
        rank_interval=(
            RationalFlatRankInterval(minimum=rank_interval[0], maximum=rank_interval[1])
            if rank_interval is not None
            else None
        ),
        symmetry_generators=(),
    )


def _modular_rank(rows: tuple[tuple[int, ...], ...], prime: int) -> int:
    if not rows:
        return 0
    vectors: list[list[int]] = []
    for row in rows:
        vector = [value % prime for value in row]
        for basis in vectors:
            pivot = next((index for index, value in enumerate(basis) if value), None)
            if pivot is not None and vector[pivot]:
                factor = vector[pivot] * pow(basis[pivot], -1, prime) % prime
                vector = [
                    (value - factor * basis_value) % prime
                    for value, basis_value in zip(vector, basis, strict=True)
                ]
        if any(vector):
            pivot = next(index for index, value in enumerate(vector) if value)
            inverse = pow(vector[pivot], -1, prime)
            vectors.append([value * inverse % prime for value in vector])
    return len(vectors)


def _modular_closed(
    rows: tuple[tuple[int, ...], ...], subset: tuple[int, ...], prime: int
) -> tuple[int, ...]:
    selected = tuple(rows[index] for index in subset)
    rank = _modular_rank(selected, prime)
    return tuple(
        index
        for index, row in enumerate(rows)
        if _modular_rank((*selected, row), prime) == rank
    )


def test_integer_configuration_distinguishes_rational_and_gf3_matroids() -> None:
    rows = ((1, 2, 3), (4, 5, 6), (7, 8, 9))

    rational = classify_clause_constrained_rational_flats(
        _rational_problem(rows, forbidden_rows=((1, 1, 1),))
    )
    modular = _run(rows, prime=3, forbidden_rows=((1, 1, 1),))

    assert rational.outcome.status == "COMPLETE_EXACT"
    assert modular.outcome.status == "COMPLETE_EXACT"
    assert tuple(
        representative.closed_candidate_indices
        for representative in rational.outcome.representatives
    ) == ((), (0,), (1,), (2,))
    assert tuple(
        representative.closed_candidate_indices
        for representative in modular.outcome.representatives
    ) == ((), (0, 1, 2))


def test_zero_rank_flat_and_modular_parallel_rows_are_labelled() -> None:
    result = _run(((0, 0), (1, 0), (2, 0), (0, 1)), prime=3)

    assert result.outcome.status == "COMPLETE_EXACT"
    assert tuple(
        representative.closed_candidate_indices
        for representative in result.outcome.representatives
    ) == ((0,), (0, 1, 2), (0, 1, 2, 3), (0, 3))
    assert result.outcome.representatives[0].rank == 0
    assert result.outcome.representatives[0].row_space_basis.vectors == ()
    assert len(result.outcome.representatives[0].annihilator_basis.vectors) == 2


def test_empty_configuration_has_one_zero_rank_flat() -> None:
    result = _run((), prime=3)

    assert result.outcome.status == "COMPLETE_EXACT"
    assert result.outcome.solution_flat_count == 1
    assert result.outcome.representatives[0].closed_candidate_indices == ()
    assert result.outcome.representatives[0].rank == 0
    assert tuple(result.outcome.representatives[0].annihilator_basis.vectors) == ((1,),)


def test_empty_clause_and_zero_forbidden_vector_make_exact_empty_family() -> None:
    empty_clause = _run(((1, 0),), prime=3, clauses=((),))
    zero_forbidden = _run(((1, 0),), prime=3, forbidden_rows=((0, 0),))

    for result in (empty_clause, zero_forbidden):
        assert result.outcome.status == "COMPLETE_EXACT"
        assert result.outcome.representatives == ()
        assert result.outcome.solution_flat_count == 0


def test_required_clause_and_rank_interval_filter_the_complete_family() -> None:
    result = _run(
        ((1, 0), (0, 1), (1, 1)),
        prime=5,
        clauses=((2,),),
        rank_interval=(1, 2),
    )

    assert result.outcome.status == "COMPLETE_EXACT"
    assert tuple(
        (representative.closed_candidate_indices, representative.rank)
        for representative in result.outcome.representatives
    ) == (((0, 1, 2), 2), ((2,), 1))


def test_coordinate_swap_identifies_modular_coordinate_lines() -> None:
    result = _run(
        ((1, 0), (0, 1)),
        prime=5,
        rank_interval=(1, 1),
        symmetry_generators=(
            PrimeFieldFlatSymmetryGenerator(
                coordinate_permutation=(1, 0), candidate_permutation=(1, 0)
            ),
        ),
    )

    assert result.outcome.status == "COMPLETE_EXACT"
    assert result.symmetry_group_order == 2
    assert result.outcome.solution_flat_count == 2
    assert len(result.outcome.representatives) == 1
    representative = result.outcome.representatives[0]
    assert representative.closed_candidate_indices == (0,)
    assert representative.orbit_size == 2
    assert representative.stabilizer_order == 1


def test_symmetry_requires_modular_row_equality_not_rational_primitivity() -> None:
    # Over QQ, (2,1) is a distinct primitive row. Over GF(3), it equals (1,2)
    # after scaling, so this paired action is valid only in the prime field.
    result = _run(
        ((1, 2), (2, 1)),
        prime=3,
        symmetry_generators=(
            PrimeFieldFlatSymmetryGenerator(
                coordinate_permutation=(1, 0), candidate_permutation=(1, 0)
            ),
        ),
    )

    assert result.outcome.status == "COMPLETE_EXACT"
    assert result.symmetry_group_order == 2
    assert result.outcome.solution_flat_count == 2
    assert len(result.outcome.representatives) == 2


def test_incompatible_symmetry_is_a_typed_domain_error() -> None:
    import pytest

    from jacobian.catalog.models import OperationDomainValidationError

    problem = _problem(
        ((1, 0), (0, 1)),
        prime=5,
        symmetry_generators=(
            PrimeFieldFlatSymmetryGenerator(
                coordinate_permutation=(1, 0), candidate_permutation=(0, 1)
            ),
        ),
    )

    with pytest.raises(OperationDomainValidationError) as error:
        classify_clause_constrained_prime_field_flats(problem)

    assert error.value.errors()[0]["type"] == "prime_field_flat.candidate_symmetry"


def test_multiple_generating_sets_deduplicate_to_one_closed_flat() -> None:
    result = _run(((1, 0), (2, 0), (3, 0), (0, 1)), prime=7)

    assert result.outcome.status == "COMPLETE_EXACT"
    assert (0, 1, 2, 3) in {
        representative.closed_candidate_indices
        for representative in result.outcome.representatives
    }


def test_small_sources_match_independent_modular_oracle() -> None:
    rows = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1))
    clauses = ((1, 3), (4, 5))
    forbidden_rows = ((1, 2, 0),)
    prime = 5
    result = _run(
        rows,
        prime=prime,
        clauses=clauses,
        forbidden_rows=forbidden_rows,
        rank_interval=(1, 2),
    )

    expected: set[tuple[int, ...]] = set()
    indices = tuple(range(len(rows)))
    for size in range(len(indices) + 1):
        for subset in combinations(indices, size):
            closed = _modular_closed(rows, subset, prime)
            if len(closed) != size:
                continue
            rank = _modular_rank(tuple(rows[index] for index in closed), prime)
            if not 1 <= rank <= 2:
                continue
            if any(set(closed).isdisjoint(clause) for clause in clauses):
                continue
            if any(
                _modular_rank(
                    (*tuple(rows[index] for index in closed), forbidden), prime
                )
                == rank
                for forbidden in forbidden_rows
            ):
                continue
            expected.add(closed)

    assert result.outcome.status == "COMPLETE_EXACT"
    assert {
        representative.closed_candidate_indices
        for representative in result.outcome.representatives
    } == expected


def test_representatives_replay_all_defining_invariants_over_gf_p() -> None:
    rows = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    prime = 5
    forbidden_rows = ((1, 1, 1),)
    result = _run(rows, prime=prime, forbidden_rows=forbidden_rows)

    assert result.outcome.status == "COMPLETE_EXACT"
    for representative in result.outcome.representatives:
        basis = representative.row_space_basis.vectors
        closed = representative.closed_candidate_indices
        for row in basis:
            assert row == tuple(value % prime for value in row)
        for row in basis:
            for annihilator in representative.annihilator_basis.vectors:
                assert (
                    sum(
                        left * right
                        for left, right in zip(row, annihilator, strict=True)
                    )
                    % prime
                    == 0
                )
        assert len(basis) == representative.rank
        for index, row in enumerate(rows):
            in_span = _modular_rank((*basis, row), prime) == representative.rank
            assert in_span == (index in closed)
        for forbidden in forbidden_rows:
            assert _modular_rank((*basis, forbidden), prime) > representative.rank


def test_catalog_example_request_and_result_round_trip() -> None:
    operation = _operation()
    request = ClauseConstrainedPrimeFieldFlatRequest.model_validate(
        operation.examples[0].input
    )
    result = operation.run(request)

    assert result.outcome.status == "COMPLETE_EXACT"
    assert (
        ClauseConstrainedPrimeFieldFlatRequest.model_validate_json(
            request.model_dump_json()
        )
        == request
    )
    assert (
        ClauseConstrainedPrimeFieldFlatClassification.model_validate_json(
            result.model_dump_json()
        )
        == result
    )


def test_serialized_complete_claim_and_representative_are_verifiable() -> None:
    result = _run(((1,),), prime=3, rank_interval=(1, 1))
    decoded = ClauseConstrainedPrimeFieldFlatClassification.model_validate_json(
        result.model_dump_json()
    )
    assert verify_prime_field_flat_classification(decoded)
    representative = decoded.outcome.representatives[0]
    assert verify_prime_field_flat_representative(decoded, representative)

    forged = result.model_dump(mode="json")
    forged["outcome"]["representatives"][0]["orbit_size"] = 2
    forged_result = ClauseConstrainedPrimeFieldFlatClassification.model_validate(forged)
    assert not verify_prime_field_flat_classification(forged_result)


def test_work_limit_returns_incomplete_not_empty_complete() -> None:
    from dataclasses import replace

    problem = _problem(((1, 0), (0, 1), (1, 1)), prime=5, clauses=((2,),))
    plan = _admit_problem(problem)
    exact_cutoff = plan.ledger.consumed + _subset_container_work(0) + 1
    plan.ledger.limit = exact_cutoff
    plan = replace(plan, search_work_limit=exact_cutoff)

    result = _classify(problem, plan)

    assert result.outcome.status == "INCOMPLETE"
    assert result.outcome.reason == "SEARCH_WORK_LIMIT"
    assert result.outcome.explored_state_orbit_count == 0
    assert result.outcome.consumed_search_work < exact_cutoff
