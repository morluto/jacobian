from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceSubset,
    PeriodicCongruenceUnionSource,
)
from jacobian.math.number_theory.periodic_prefix_count.operations import (
    compute_periodic_union_prefix_count,
    verify_periodic_union_prefix_count,
)


def _source(
    subsets: Sequence[tuple[int, Sequence[int]]], complement: bool = False
) -> PeriodicCongruenceUnionSource:
    return PeriodicCongruenceUnionSource(
        subsets=tuple(
            PeriodicCongruenceSubset(modulus=m, residues=tuple(r)) for m, r in subsets
        ),
        complement=complement,
    )


def test_fixture_mod2_or_mod3() -> None:
    """On [1,6], the union of 0 mod 2 and 1 mod 3 is {1,2,4,6}, count 4."""
    source = _source([(2, [0]), (3, [1])])
    result = compute_periodic_union_prefix_count(source, 6)
    assert result.count == 4
    assert result.occupied_count == 4  # period 6: residues 0,1,2,4


def test_empty_union() -> None:
    """Empty union has count 0."""
    source = PeriodicCongruenceUnionSource(subsets=(), complement=False)
    result = compute_periodic_union_prefix_count(source, 10)
    assert result.count == 0


def test_mod1_all_integers() -> None:
    """0 mod 1: all positive integers."""
    source = _source([(1, [0])])
    result = compute_periodic_union_prefix_count(source, 10)
    assert result.count == 10


def test_cutoff_zero() -> None:
    """Cutoff 0: count 0."""
    source = _source([(2, [0])])
    result = compute_periodic_union_prefix_count(source, 0)
    assert result.count == 0


def test_periodicity() -> None:
    """Extending the cutoff by one period increases the count by occupied_count."""
    source = _source([(2, [0])])
    result_small = compute_periodic_union_prefix_count(source, 10)
    result_large = compute_periodic_union_prefix_count(source, 12)
    assert int(result_large.count) - int(result_small.count) == 1


def test_result_preserves_source() -> None:
    """Result retains the source and cutoff."""
    source = _source([(2, [0])])
    result = compute_periodic_union_prefix_count(source, 10)
    assert result.source == source
    assert result.cutoff == 10


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("common_period", "12"),
        ("occupied_count", "5"),
        ("count", "5"),
    ],
)
def test_serialized_result_decoding_is_structural_and_verification_is_explicit(
    field: str, forged_value: str
) -> None:
    source = _source([(2, [0]), (3, [1])])
    result = compute_periodic_union_prefix_count(source, 6)
    payload = json.loads(result.model_dump_json())
    payload[field] = forged_value

    claim = type(result).model_validate_json(json.dumps(payload))

    assert getattr(claim, field) == int(forged_value)
    assert not verify_periodic_union_prefix_count(claim)


def test_complemented_large_period_uses_scalar_count() -> None:
    """A large period is counted without constructing its residue profile."""
    source = _source([(100000, [0])], complement=True)
    result = compute_periodic_union_prefix_count(source, 100000)
    assert result.count == 99999
    assert result.occupied_count == 99999


def test_negative_cutoff_is_a_typed_domain_rejection() -> None:
    source = _source([(2, [0])])
    with pytest.raises(OperationDomainValidationError, match="nonnegative"):
        compute_periodic_union_prefix_count(source, -1)


def test_scalar_count_keeps_period_lift_plan() -> None:
    """A dense, small-period source is counted without IE merge limits."""
    source = _source(
        [(1000, list(range(999)))],
        complement=False,
    )
    result = compute_periodic_union_prefix_count(source, 1000)
    assert result.count == 999


def test_cutoff_digit_bound_is_typed() -> None:
    source = _source([(2, [0])])
    with pytest.raises(OperationDomainValidationError, match="at most 32768 digits"):
        compute_periodic_union_prefix_count(source, 10**32768)


def test_scalar_cutoff_can_exceed_period_digit_bound() -> None:
    source = PeriodicCongruenceUnionSource(subsets=(), complement=False)
    result = compute_periodic_union_prefix_count(source, 10**256)
    assert result.count == 0
