"""Regression coverage for maximum binary finite-set outputs."""

import pytest

from jacobian.math.finite_sets._models import (
    FiniteIntegerSet,
    FiniteSetCardinalityResult,
    FiniteSetCoverageRequest,
    FiniteSetElementListResult,
    FiniteSetPairRequest,
)
from jacobian.math.finite_sets._operations import (
    decide_exact_cover,
    set_symmetric_difference,
    set_union,
    union_cardinality,
)


def test_finite_set_operations_support_canonical_integers_above_python_digit_limit() -> (
    None
):
    value = "1" + "0" * 4_300

    coverage = decide_exact_cover(
        FiniteSetCoverageRequest(
            scope=FiniteIntegerSet(elements=(value,)),
            values=(),
        )
    )
    union = set_union(
        FiniteSetPairRequest(
            left=FiniteIntegerSet(elements=(value,)),
            right=FiniteIntegerSet(elements=()),
        )
    )

    assert coverage.missing == (value,)
    assert union.elements == (value,)


@pytest.fixture
def maximum_disjoint_pair() -> FiniteSetPairRequest:
    return FiniteSetPairRequest(
        left=FiniteIntegerSet(elements=tuple(str(value) for value in range(128))),
        right=FiniteIntegerSet(elements=tuple(str(value) for value in range(128, 256))),
    )


def test_maximum_disjoint_union_fits_public_result_contract(
    maximum_disjoint_pair: FiniteSetPairRequest,
) -> None:
    result = set_union(maximum_disjoint_pair)

    assert isinstance(result, FiniteSetElementListResult)
    assert len(result.elements) == 256
    assert result.elements == tuple(str(value) for value in range(256))


def test_maximum_disjoint_symmetric_difference_fits_public_result_contract(
    maximum_disjoint_pair: FiniteSetPairRequest,
) -> None:
    result = set_symmetric_difference(maximum_disjoint_pair)

    assert isinstance(result, FiniteSetElementListResult)
    assert len(result.elements) == 256
    assert result.elements == tuple(str(value) for value in range(256))


def test_maximum_disjoint_union_cardinality_fits_public_result_contract(
    maximum_disjoint_pair: FiniteSetPairRequest,
) -> None:
    result = union_cardinality(maximum_disjoint_pair)

    assert isinstance(result, FiniteSetCardinalityResult)
    assert result.cardinality == 256
