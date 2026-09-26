"""Exact tests for weighted independent-set optimization."""

from __future__ import annotations

from itertools import combinations, product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
)
from jacobian.math.combinatorics.matroids import (
    LinearMatroid,
    MatroidWeightFunction,
    maximum_weight_independent_set_result,
    verify_maximum_weight_independent_set,
)
from jacobian.math.combinatorics.matroids._models import (
    MAX_SPLIT_WEIGHT_DIGITS,
    MaximumWeightIndependentSetRequest,
    MaximumWeightIndependentSetResult,
)
from jacobian.math.combinatorics.matroids._tools import (
    TOOLS,
    _run_maximum_weight_independent_set,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def _matroid(prime: int, rows: tuple[tuple[int, ...], ...]) -> LinearMatroid:
    return LinearMatroid(
        matrix=PrimeFieldMatrix(prime=prime, entries=rows, columns=len(rows[0]))
    )


def _weights(matroid: LinearMatroid, values: tuple[int, ...]) -> MatroidWeightFunction:
    return MatroidWeightFunction(ground_axis=matroid.ground_axis, values=values)


def _independent_by_linear_relation(
    matroid: LinearMatroid, subset: tuple[int, ...]
) -> bool:
    """Small independent oracle: exhaust all nonzero coefficient vectors."""
    p = matroid.matrix.prime
    for coeffs in product(range(p), repeat=len(subset)):
        if not any(coeffs):
            continue
        if all(
            sum(coeffs[k] * row[index] for k, index in enumerate(subset)) % p == 0
            for row in matroid.matrix.entries
        ):
            return False
    return True


def _brute_force_optimum(matroid: LinearMatroid, weights: tuple[int, ...]) -> int:
    best = 0  # The empty independent set is always feasible.
    for size in range(matroid.ground_size + 1):
        for subset in combinations(range(matroid.ground_size), size):
            if _independent_by_linear_relation(matroid, subset):
                best = max(best, sum(weights[i] for i in subset))
    return best


def test_negative_weights_do_not_force_a_basis() -> None:
    matroid = _matroid(5, ((1, 0, 1), (0, 1, 1)))
    result = maximum_weight_independent_set_result(
        matroid, _weights(matroid, (4, -3, 2))
    )

    assert result.independent_set == (0, 2)
    assert result.total_weight == 6
    assert result.rank == 2
    assert result.greedy_order == (0, 2)
    assert result != maximum_weight_independent_set_result(
        matroid, _weights(matroid, (-1, -3, 0))
    )
    all_nonpositive = maximum_weight_independent_set_result(
        matroid, _weights(matroid, (-1, -3, 0))
    )
    assert all_nonpositive.independent_set == ()
    assert all_nonpositive.total_weight == all_nonpositive.rank == 0


def test_loop_zero_ties_and_degenerate_ground() -> None:
    loop_matroid = _matroid(3, ((1, 0, 0), (0, 1, 0)))
    result = maximum_weight_independent_set_result(
        loop_matroid, _weights(loop_matroid, (5, 5, 100))
    )
    assert result.independent_set == (0, 1)
    assert result.total_weight == 10
    assert 2 not in result.independent_set
    assert result.greedy_order == (2, 0, 1)

    empty = LinearMatroid(matrix=PrimeFieldMatrix(prime=2, entries=((),), columns=0))
    assert (
        maximum_weight_independent_set_result(
            empty, _weights(empty, ())
        ).independent_set
        == ()
    )


def test_greedy_matches_independent_coefficient_and_subset_enumeration() -> None:
    fixtures = (
        (_matroid(2, ((1, 0, 1, 1), (0, 1, 1, 0))), (4, -2, 4, 1)),
        (_matroid(3, ((1, 0, 1, 1), (0, 1, 1, 2))), (0, 3, 2, 3)),
        (_matroid(2, ((0, 0, 0),)), (-1, 5, 0)),
    )
    for matroid, weights in fixtures:
        result = maximum_weight_independent_set_result(
            matroid, _weights(matroid, weights)
        )
        assert result.total_weight == _brute_force_optimum(matroid, weights)
        assert _independent_by_linear_relation(matroid, result.independent_set)
        assert result.rank == len(result.independent_set)


def test_row_operations_preserve_optimum() -> None:
    matroid = _matroid(5, ((1, 0, 1, 1), (0, 1, 1, 2)))
    # Invertible row operation R1 <- R1 + 2*R2 over GF(5).
    transformed = _matroid(5, ((1, 2, 3, 0), (0, 1, 1, 2)))
    weights = (4, -1, 3, 3)
    left = maximum_weight_independent_set_result(matroid, _weights(matroid, weights))
    right = maximum_weight_independent_set_result(
        transformed, _weights(transformed, weights)
    )
    assert (left.independent_set, left.total_weight) == (
        right.independent_set,
        right.total_weight,
    )


def test_forged_claims_fail_independent_greedy_replay() -> None:
    matroid = _matroid(5, ((1, 0, 1), (0, 1, 1)))
    result = maximum_weight_independent_set_result(
        matroid, _weights(matroid, (4, 2, 3))
    )
    assert verify_maximum_weight_independent_set(result)

    feasible_but_suboptimal = MaximumWeightIndependentSetResult.model_construct(
        matroid=result.matroid,
        weight_function=result.weight_function,
        independent_set=(1,),
        total_weight=2,
        rank=1,
        greedy_order=result.greedy_order,
    )
    assert not verify_maximum_weight_independent_set(feasible_but_suboptimal)


def test_request_bounds_and_native_parity() -> None:
    matroid = _matroid(5, ((1, 0, 1), (0, 1, 1)))
    with pytest.raises(ValidationError):
        MaximumWeightIndependentSetRequest(
            matroid=matroid,
            weight_function=MatroidWeightFunction(
                ground_axis=("0", "1"), values=(1, 2)
            ),
        )
    with pytest.raises(OperationDomainValidationError):
        maximum_weight_independent_set_result(
            matroid,
            MatroidWeightFunction(ground_axis=("x", "y", "z"), values=(1, 2, 3)),
        )

    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "matroid.independent_set.maximum_weight.compute"
    )
    request = MaximumWeightIndependentSetRequest.model_validate(tool.examples[0].input)
    assert _run_maximum_weight_independent_set(request) == tool.run(request)
    assert tool.run(request).independent_set == (0, 2)


def test_weight_digit_and_ground_bound_edges() -> None:
    one_loop = _matroid(2, ((0,),))
    accepted_weights = MatroidWeightFunction(
        ground_axis=one_loop.ground_axis, values=(10**12 - 1,)
    )
    accepted = maximum_weight_independent_set_result(one_loop, accepted_weights)
    assert accepted.independent_set == ()
    wider_split_carrier = MatroidWeightFunction(
        ground_axis=one_loop.ground_axis, values=(10**12,)
    )
    with pytest.raises(OperationDomainValidationError, match="decimal digits"):
        maximum_weight_independent_set_result(one_loop, wider_split_carrier)
    with pytest.raises(ValidationError, match="decimal digits"):
        MatroidWeightFunction(
            ground_axis=one_loop.ground_axis, values=(10 ** (MAX_SPLIT_WEIGHT_DIGITS),)
        )

    max_ground = _matroid(2, ((0,) * 256,))
    result = maximum_weight_independent_set_result(
        max_ground, _weights(max_ground, (1,) * 256)
    )
    assert result.independent_set == ()
    with pytest.raises(ValidationError):
        MatroidWeightFunction(ground_axis=max_ground.ground_axis, values=(1,) * 257)


def test_retained_labels_are_admitted_before_the_rank_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The result retains both the source matroid axis and the canonicalized
    # weight-function axis, so a native operand with unbounded label text is
    # refused on its allocation before any independence probe runs.
    import jacobian.catalog.models as catalog_models
    import jacobian.math.combinatorics.matroids.operations as operations
    from jacobian.math.combinatorics.matroids._models import (
        MAX_GROUND_AXIS_CODEPOINTS,
    )

    huge_label = "x" * (MAX_GROUND_AXIS_CODEPOINTS + 1)
    matroid = LinearMatroid(
        matrix=PrimeFieldMatrix(prime=2, entries=((1,),), columns=1),
        ground_labels=(huge_label,),
    )
    weight_function = MatroidWeightFunction(ground_axis=(huge_label,), values=(1,))

    def unexpected(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("rank kernel ran before retained-output admission")

    monkeypatch.setattr(operations, "pf_rank", unexpected)
    with pytest.raises(catalog_models.OperationResourceAdmissionError) as error:
        maximum_weight_independent_set_result(matroid, weight_function)
    assert (
        error.value.errors()[0]["type"]
        == "matroid.maximum_weight_independent_set.work_bound"
    )


def test_axis_inside_the_allocation_bound_returns_the_exact_optimum() -> None:
    from jacobian.math.combinatorics.matroids._models import (
        MAX_GROUND_AXIS_CODEPOINTS,
    )

    first_label = "a" * 3
    second_label = "b" * (MAX_GROUND_AXIS_CODEPOINTS - 3)
    matroid = LinearMatroid(
        matrix=PrimeFieldMatrix(prime=2, entries=((1, 0), (0, 1)), columns=2),
        ground_labels=(first_label, second_label),
    )
    weight_function = MatroidWeightFunction(
        ground_axis=(first_label, second_label), values=(1, 2)
    )
    result = maximum_weight_independent_set_result(matroid, weight_function)
    assert result.independent_set == (0, 1)
    assert result.total_weight == 3
    assert verify_maximum_weight_independent_set(result)
