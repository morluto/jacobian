"""Regression coverage for maximum binary finite-set outputs."""

import pytest

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.sets._models import (
    MAX_FINITE_INTEGER_SET_ELEMENTS,
    FiniteIntegerSet,
    FiniteSetCardinalityResult,
    FiniteSetCoverageRequest,
    FiniteSetElementListResult,
    FiniteSetPairRequest,
)
from jacobian.math.combinatorics.finite_structures.sets._tools import (
    decide_exact_cover,
    set_symmetric_difference,
    set_union,
    union_cardinality,
)


def test_finite_set_operations_support_canonical_integers_above_python_digit_limit() -> (
    None
):
    value = "1" + "0" * 4_300

    native_value = parse_canonical_integer(value)
    coverage = decide_exact_cover(
        FiniteSetCoverageRequest(
            scope=FiniteIntegerSet.model_validate_json(
                '{"elements":["' + value + '"]}'
            ),
            values=(),
        )
    )
    union = set_union(
        FiniteSetPairRequest(
            left=FiniteIntegerSet(elements=(native_value,)),
            right=FiniteIntegerSet(elements=()),
        )
    )

    assert coverage.missing == (native_value,)
    assert union.elements == (native_value,)


@pytest.fixture
def maximum_disjoint_pair() -> FiniteSetPairRequest:
    return FiniteSetPairRequest(
        left=FiniteIntegerSet(elements=tuple(value for value in range(128))),
        right=FiniteIntegerSet(elements=tuple(value for value in range(128, 256))),
    )


def test_maximum_disjoint_union_fits_public_result_contract(
    maximum_disjoint_pair: FiniteSetPairRequest,
) -> None:
    result = set_union(maximum_disjoint_pair)

    assert isinstance(result, FiniteSetElementListResult)
    assert len(result.elements) == 256
    assert result.elements == tuple(value for value in range(256))


def test_maximum_disjoint_symmetric_difference_fits_public_result_contract(
    maximum_disjoint_pair: FiniteSetPairRequest,
) -> None:
    result = set_symmetric_difference(maximum_disjoint_pair)

    assert isinstance(result, FiniteSetElementListResult)
    assert len(result.elements) == 256
    assert result.elements == tuple(value for value in range(256))


def test_maximum_disjoint_union_cardinality_fits_public_result_contract(
    maximum_disjoint_pair: FiniteSetPairRequest,
) -> None:
    result = union_cardinality(maximum_disjoint_pair)

    assert isinstance(result, FiniteSetCardinalityResult)
    assert result.cardinality == 256


def test_union_cardinality_rejects_a_result_larger_than_its_contract() -> None:
    request = FiniteSetPairRequest(
        left=FiniteIntegerSet(elements=tuple(range(MAX_FINITE_INTEGER_SET_ELEMENTS))),
        right=FiniteIntegerSet(elements=(MAX_FINITE_INTEGER_SET_ELEMENTS,)),
    )

    with pytest.raises(OperationDomainValidationError) as exc_info:
        union_cardinality(request)

    assert exc_info.value.errors()[0]["type"] == "finite_set.result_size_exceeded"
