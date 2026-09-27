"""Tests for matroid.basis.maximum_weight.compute."""

from __future__ import annotations

from itertools import combinations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.matroids._models import (
    LinearMatroid,
    MatroidWeightFunction,
    MaximumWeightBasisRequest,
    MaximumWeightBasisResult,
)
from jacobian.math.combinatorics.matroids._tools import _run_maximum_weight_basis
from jacobian.math.combinatorics.matroids.operations import (
    matroid_rank,
    maximum_weight_basis_result,
    verify_maximum_weight_basis,
)
from jacobian.math.matrices.finite_fields.linear_algebra import (
    PrimeFieldMatrix,
)
from jacobian.math.matrices.finite_fields.linear_algebra import (
    rank as pf_rank,
)


def _matroid(prime: int, rows: tuple[tuple[int, ...], ...]) -> LinearMatroid:
    columns = len(rows[0]) if rows else 0
    return LinearMatroid(
        matrix=PrimeFieldMatrix(prime=prime, entries=rows, columns=columns)
    )


def _triangle() -> LinearMatroid:
    return _matroid(5, ((1, 0, 1), (0, 1, 1)))


def _solve(matroid: LinearMatroid, weights: tuple[int, ...]):
    weight_function = MatroidWeightFunction.model_construct(
        ground_axis=matroid.ground_axis, values=weights
    )
    return maximum_weight_basis_result(matroid, weight_function)


def _brute_force_optimum(matroid: LinearMatroid, weights: tuple[int, ...]) -> int:
    n = matroid.ground_size
    rank = matroid_rank(matroid)
    best: int | None = None
    for combo in combinations(range(n), rank):
        cols = [tuple(row[j] for j in combo) for row in matroid.matrix.entries]
        selected = PrimeFieldMatrix(
            prime=matroid.matrix.prime, entries=tuple(cols), columns=len(combo)
        )
        if _oracle_rank(selected) == rank:
            total = sum(weights[j] for j in combo)
            best = total if best is None else max(best, total)
    assert best is not None
    return best


def _oracle_rank(matrix: PrimeFieldMatrix) -> int:
    """Small independent modular elimination oracle for exhaustive fixtures."""
    rows = [list(row) for row in matrix.entries]
    pivot_row = 0
    for column in range(matrix.columns):
        pivot = next(
            (r for r in range(pivot_row, len(rows)) if rows[r][column] % matrix.prime),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        inverse = pow(rows[pivot_row][column] % matrix.prime, -1, matrix.prime)
        rows[pivot_row] = [
            (value * inverse) % matrix.prime for value in rows[pivot_row]
        ]
        for r in range(len(rows)):
            if r != pivot_row:
                factor = rows[r][column] % matrix.prime
                rows[r] = [
                    (left - factor * right) % matrix.prime
                    for left, right in zip(rows[r], rows[pivot_row], strict=True)
                ]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row


def test_known_answer_triangle() -> None:
    """Weights (3, 2, 5) on the rank-2 triangle pick {0, 2} with total 8."""
    result = _solve(_triangle(), (3, 2, 5))
    assert result.basis == (0, 2)
    assert result.total_weight == 8
    assert result.rank == 2
    assert result.greedy_order == (2, 0, 1)
    assert len(result.exchange_ledger) == 1
    row = result.exchange_ledger[0]
    assert row.outside == 1
    assert row.circuit == (0, 1, 2)
    assert row.best_delta == 2 - 3


def test_tied_weights_break_ties_by_index() -> None:
    """Uniform U(2,3) with equal weights deterministically picks {0, 1}."""
    result = _solve(_triangle(), (4, 4, 4))
    assert result.basis == (0, 1)
    assert result.total_weight == 8
    assert result.greedy_order == (0, 1, 2)


def test_negative_weights_still_return_a_basis() -> None:
    """A basis has cardinality r(M) even when every weight is negative."""
    result = _solve(_triangle(), (-1, -5, -2))
    assert result.basis == (0, 2)
    assert result.total_weight == -3
    assert result.rank == 2


def test_zero_column_loop_never_selected() -> None:
    """A loop (zero column) is never part of any basis."""
    matroid = _matroid(5, ((1, 0, 0), (0, 1, 0)))
    result = _solve(matroid, (1, 1, 100))
    assert 2 not in result.basis
    assert result.basis == (0, 1)
    assert result.total_weight == 2


def test_graphic_cycle_maximum_spanning_tree() -> None:
    """Reduced incidence over GF(2) of C4: max-weight basis is a max tree."""
    # C4 on vertices 0..3, edges e0=(0,1), e1=(1,2), e2=(2,3), e3=(0,3);
    # rows are vertices 0..2 of the GF(2) incidence matrix.
    matroid = _matroid(
        2,
        (
            (1, 0, 0, 1),
            (1, 1, 0, 0),
            (0, 1, 1, 0),
        ),
    )
    assert matroid_rank(matroid) == 3
    result = _solve(matroid, (1, 2, 3, 4))
    assert result.basis == (1, 2, 3)
    assert result.total_weight == 9


def test_degenerate_rank_zero_matroid() -> None:
    """A single loop has the empty basis with total weight zero."""
    result = _solve(_matroid(5, ((0,),)), (7,))
    assert result.basis == ()
    assert result.total_weight == 0
    assert result.rank == 0
    assert result.exchange_ledger[0].circuit == (0,)


def test_weight_length_mismatch_rejected() -> None:
    """Weights must cover the ground set exactly once."""
    with pytest.raises(OperationDomainValidationError):
        _solve(_triangle(), (1, 2))


def test_non_integer_weight_rejected() -> None:
    """Weights must be exact integers."""
    with pytest.raises(OperationDomainValidationError):
        _solve(_triangle(), (1.5, 2, 3))  # type: ignore[arg-type]


def test_greedy_order_replay_invariant() -> None:
    """Replaying the greedy order reproduces the claimed basis and total."""
    matroid = _matroid(5, ((1, 0, 1, 1), (0, 1, 1, 2)))
    weights = (5, 1, 4, 4)
    result = _solve(matroid, weights)
    assert tuple(sorted(result.greedy_order)) == (0, 1, 2, 3)
    assert [weights[e] for e in result.greedy_order] == sorted(
        [weights[e] for e in result.greedy_order], reverse=True
    )
    # Independent replay: walk the order, keep rank-increasing elements.
    replay: list[int] = []
    for element in result.greedy_order:
        candidate = [*replay, element]
        cols = [tuple(row[j] for j in candidate) for row in matroid.matrix.entries]
        selected = PrimeFieldMatrix(
            prime=matroid.matrix.prime, entries=tuple(cols), columns=len(candidate)
        )
        if pf_rank(selected) == len(candidate):
            replay.append(element)
    assert tuple(sorted(replay)) == result.basis
    assert sum(weights[e] for e in replay) == result.total_weight


def test_brute_force_optimality_small_cases() -> None:
    """Greedy totals match exhaustive basis enumeration on small fixtures."""
    fixtures = [
        (_matroid(5, ((1, 0, 1, 1), (0, 1, 1, 2))), (5, 1, 4, 4)),
        (_matroid(2, ((1, 1, 0, 1), (0, 1, 1, 1))), (2, 7, 3, 7)),
        (_triangle(), (0, 0, 0)),
    ]
    for matroid, weights in fixtures:
        result = _solve(matroid, weights)
        assert result.total_weight == _brute_force_optimum(matroid, weights)
        for row in result.exchange_ledger:
            assert row.best_delta <= 0


def test_no_valid_exchange_improves_total() -> None:
    """Every ledger row certifies local nonimprovement across the exchange graph."""
    matroid = _matroid(5, ((1, 0, 1, 1), (0, 1, 1, 2)))
    weights = (5, 1, 4, 4)
    result = _solve(matroid, weights)
    basis_set = set(result.basis)
    for row in result.exchange_ledger:
        assert row.outside not in basis_set
        assert row.outside in row.circuit
        for member in row.circuit:
            if member == row.outside:
                continue
            assert member in basis_set
            assert weights[row.outside] - weights[member] <= 0


def test_adversarial_mutated_basis_fails_verify() -> None:
    """Dropping a basis element breaks cardinality and fails verification."""
    result = _solve(_triangle(), (3, 2, 5))
    mutated = MaximumWeightBasisResult.model_construct(
        matroid=result.matroid,
        weight_function=result.weight_function,
        basis=(0,),
        total_weight=3,
        rank=result.rank,
        greedy_order=result.greedy_order,
        exchange_ledger=result.exchange_ledger,
    )
    assert not verify_maximum_weight_basis(mutated)


def test_adversarial_suboptimal_claim_fails_verify() -> None:
    """A feasible but suboptimal basis is rejected by the greedy replay."""
    result = _solve(_triangle(), (3, 2, 5))
    mutated = MaximumWeightBasisResult.model_construct(
        matroid=result.matroid,
        weight_function=result.weight_function,
        basis=(0, 1),
        total_weight=5,
        rank=result.rank,
        greedy_order=result.greedy_order,
        exchange_ledger=result.exchange_ledger,
    )
    assert not verify_maximum_weight_basis(mutated)


def test_verify_accepts_true_claim() -> None:
    """The kernel output verifies against itself."""
    result = _solve(_triangle(), (3, 2, 5))
    assert verify_maximum_weight_basis(result)


def test_native_vs_catalog_parity() -> None:
    """The catalog wrapper runs the same admitted kernel as the native call."""
    request = MaximumWeightBasisRequest(
        matroid=_triangle(),
        weight_function=MatroidWeightFunction(
            ground_axis=("0", "1", "2"), values=(3, 2, 5)
        ),
    )
    assert _run_maximum_weight_basis(request) == maximum_weight_basis_result(
        request.matroid, request.weight_function
    )


def test_catalog_example_input_executes() -> None:
    """The published operation example is admitted and optimal."""
    from jacobian.math.combinatorics.matroids._tools import TOOLS

    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "matroid.basis.maximum_weight.compute"
    )
    assert tool.examples
    request = MaximumWeightBasisRequest.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert result.basis == (0, 2)
    assert result.total_weight == 8


def test_ground_axis_weight_table_is_order_independent() -> None:
    """Reordered keyed rows bind by label and results normalize to source axis."""
    matroid = LinearMatroid(
        matrix=_triangle().matrix,
        ground_labels=("edge-c", "edge-a", "edge-b"),
    )
    weights = MatroidWeightFunction(
        ground_axis=("edge-b", "edge-c", "edge-a"), values=(5, 3, 2)
    )
    result = maximum_weight_basis_result(matroid, weights)
    assert result.basis == (0, 2)
    assert result.total_weight == 8
    assert result.weight_function.ground_axis == matroid.ground_axis
    assert result.weight_function.values == (3, 2, 5)


def test_weight_table_with_wrong_ground_labels_rejected() -> None:
    """Equal cardinality does not make a foreign keyed weight domain valid."""
    matroid = LinearMatroid(
        matrix=_triangle().matrix,
        ground_labels=("edge-a", "edge-b", "edge-c"),
    )
    foreign = MatroidWeightFunction(
        ground_axis=("other-a", "other-b", "other-c"), values=(3, 2, 5)
    )
    with pytest.raises(OperationDomainValidationError):
        maximum_weight_basis_result(matroid, foreign)
    with pytest.raises(ValueError):
        MaximumWeightBasisRequest(matroid=matroid, weight_function=foreign)
