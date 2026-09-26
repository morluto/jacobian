"""Exact disjoint-ground direct sums of finite delta-matroids."""

from __future__ import annotations

import pytest

from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids.delta._tools import _run_direct_sum
from jacobian.math.combinatorics.matroids.delta.extra import (
    DeltaMatroidDirectSumRequest,
)
from jacobian.math.combinatorics.matroids.delta.operations import direct_sum
from jacobian.math.combinatorics.matroids.delta.values import (
    FiniteDeltaMatroid,
    first_symmetric_exchange_obstruction,
)


def test_direct_sum_matches_independent_pairwise_union_oracle() -> None:
    left = FiniteDeltaMatroid(ground=("a", "b"), feasible=((), (0,), (0, 1), (1,)))
    right = FiniteDeltaMatroid(ground=("c",), feasible=((), (0,)))

    result = direct_sum(left, right)

    expected = tuple(
        sorted(
            tuple(sorted((*left_row, *(len(left.ground) + i for i in right_row))))
            for left_row in left.feasible
            for right_row in right.feasible
        )
    )
    assert result.direct_sum.ground == ("a", "b", "c")
    assert result.direct_sum.feasible == expected
    assert result.left_injection == (0, 1)
    assert result.right_injection == (2,)
    # The product family is itself a delta-matroid (here the complete cube).
    assert len(result.direct_sum.feasible) == 8


def test_direct_sum_empty_ground_is_identity_on_feasible_rows() -> None:
    identity = FiniteDeltaMatroid(ground=(), feasible=((),))
    value = FiniteDeltaMatroid(ground=("x",), feasible=((), (0,)))
    result = direct_sum(identity, value)
    assert result.direct_sum == value
    assert result.left_injection == ()
    assert result.right_injection == (0,)


def test_overlapping_labels_rejected_instead_of_silently_tagged() -> None:
    left = FiniteDeltaMatroid(ground=("x",), feasible=((), (0,)))
    right = FiniteDeltaMatroid(ground=("x",), feasible=((), (0,)))
    with pytest.raises(ValueError, match="disjoint"):
        _run_direct_sum(DeltaMatroidDirectSumRequest(left=left, right=right))


def test_catalog_path_returns_typed_result() -> None:
    left = FiniteDeltaMatroid(ground=("a",), feasible=((), (0,)))
    right = FiniteDeltaMatroid(ground=("b",), feasible=((),))
    result = _run_direct_sum(DeltaMatroidDirectSumRequest(left=left, right=right))
    assert result.direct_sum.feasible == ((), (0,))


def _boolean_lattice(order: int, offset: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(
            tuple(offset + bit for bit in range(order) if (index >> bit) & 1)
            for index in range(1 << order)
        )
    )


def test_direct_sum_of_complete_families_is_admitted_without_axiom_replay() -> None:
    # Two complete four-element families have 256 feasible pairs and 1,024
    # memberships: the product construction and every retained cardinality
    # envelope fit, so the direct-sum theorem must carry symmetric exchange
    # instead of charging a hypothetical recognition replay.
    left = FiniteDeltaMatroid(
        ground=("a", "b", "c", "d"),
        feasible=_boolean_lattice(4, 0),
    )
    right = FiniteDeltaMatroid(
        ground=("e", "f", "g", "h"),
        feasible=_boolean_lattice(4, 0),
    )

    result = direct_sum(left, right)

    assert result.direct_sum.ground == ("a", "b", "c", "d", "e", "f", "g", "h")
    assert len(result.direct_sum.feasible) == 256
    # Independent oracle: the product of complete families is the complete
    # eight-element family.
    expected = _boolean_lattice(8, 0)
    assert result.direct_sum.feasible == expected
    assert (
        first_symmetric_exchange_obstruction(
            FiniteFeasibleSetSystem(
                ground=result.direct_sum.ground,
                feasible=result.direct_sum.feasible,
            )
        )
        is None
    )


def test_native_direct_sum_validates_operands_before_dereferencing() -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    right = FiniteDeltaMatroid(ground=("b",), feasible=((), (0,)))
    with pytest.raises(OperationDomainValidationError) as error:
        direct_sum(object(), right)  # type: ignore[arg-type]
    assert error.value.errors()[0]["type"] == "delta_matroid.source_not_valid"

    forged_left = FiniteDeltaMatroid.model_construct(ground=("a",), feasible=((), (5,)))
    with pytest.raises(OperationDomainValidationError) as error:
        direct_sum(forged_left, right)
    assert error.value.errors()[0]["type"] == "delta_matroid.source_not_valid"

    forged_right = FiniteDeltaMatroid.model_construct(ground=("b",), feasible=())
    with pytest.raises(OperationDomainValidationError) as error:
        direct_sum(right, forged_right)
    assert error.value.errors()[0]["type"] == "delta_matroid.source_not_valid"


def test_catalog_direct_sum_reports_domain_error_for_forged_operands() -> None:
    from jacobian.catalog.models import (
        OperationDomainValidationError,
        OperationResourceAdmissionError,
    )

    right = FiniteDeltaMatroid(ground=("b",), feasible=((), (0,)))
    forged_left = FiniteDeltaMatroid.model_construct(ground=("a",), feasible=((), (9,)))
    request = DeltaMatroidDirectSumRequest.model_construct(
        left=forged_left, right=right
    )
    with pytest.raises(
        (OperationDomainValidationError, OperationResourceAdmissionError)
    ) as error:
        _run_direct_sum(request)
    assert error.value.errors()[0]["type"] == "delta_matroid.source_not_valid"
    assert error.value.errors()[0]["loc"][0] == "left"
