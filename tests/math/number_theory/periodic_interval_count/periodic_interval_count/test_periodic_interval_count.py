from __future__ import annotations

from collections.abc import Sequence

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceSubset,
    PeriodicCongruenceUnionSource,
)
from jacobian.math.number_theory.periodic_interval_count.operations import (
    compute_periodic_interval_count,
)


def _source(
    modulus: int, residues: Sequence[int], complement: bool = False
) -> PeriodicCongruenceUnionSource:
    return PeriodicCongruenceUnionSource(
        subsets=(
            PeriodicCongruenceSubset(
                modulus=str(modulus), residues=tuple(str(r) for r in residues)
            ),
        ),
        complement=complement,
    )


def test_multiples_of_3() -> None:
    source = _source(3, [0])
    result = compute_periodic_interval_count(source, 1, 20)
    # 3, 6, 9, 12, 15, 18 -> 6
    assert result.count == "6"


def test_complement() -> None:
    source = _source(3, [0], complement=True)
    result = compute_periodic_interval_count(source, 1, 10)
    # Non-multiples of 3 in [1,10]: 1,2,4,5,7,8,10 -> 7
    assert result.count == "7"


def test_empty_interval() -> None:
    source = _source(3, [0])
    result = compute_periodic_interval_count(source, 10, 5)
    assert result.count == "0"


def test_result_preserves_source() -> None:
    source = _source(5, [0])
    result = compute_periodic_interval_count(source, 1, 10)
    assert result.source == source
    assert result.lower == "1"
    assert result.upper == "10"


def test_huge_interval_uses_periodic_arithmetic() -> None:
    source = _source(3, [0])
    result = compute_periodic_interval_count(source, 1, 10**18)

    assert result.count == str(10**18 // 3)


def test_negative_interval_uses_floor_period_blocks() -> None:
    source = _source(3, [0])
    result = compute_periodic_interval_count(source, -7, 2)

    assert result.count == "3"


def test_large_period_complement_uses_scalar_rank() -> None:
    source = _source(1_000_000, [0], complement=True)

    result = compute_periodic_interval_count(source, -2, 2)

    assert result.count == "4"


def test_endpoint_pair_must_fit_the_result_envelope() -> None:
    source = _source(1, [0])
    endpoint = 10**3_600_000

    with pytest.raises(OperationDomainValidationError, match="output budget"):
        compute_periodic_interval_count(source, -endpoint, endpoint)
