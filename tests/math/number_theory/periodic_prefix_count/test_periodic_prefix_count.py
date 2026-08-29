from __future__ import annotations

from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceSubset,
    PeriodicCongruenceUnionSource,
)
from jacobian.math.number_theory.periodic_prefix_count.operations import (
    compute_periodic_union_prefix_count,
)


def _source(subsets, complement=False):
    return PeriodicCongruenceUnionSource(
        subsets=tuple(
            PeriodicCongruenceSubset(modulus=m, residues=tuple(r)) for m, r in subsets
        ),
        complement=complement,
    )


def test_fixture_mod2_or_mod3() -> None:
    """On [1,6], the union of 0 mod 2 and 1 mod 3 is {1,2,4,6}, count 4."""
    source = _source([("2", ["0"]), ("3", ["1"])])
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
    source = _source([("1", ["0"])])
    result = compute_periodic_union_prefix_count(source, 10)
    assert result.count == 10


def test_cutoff_zero() -> None:
    """Cutoff 0: count 0."""
    source = _source([("2", ["0"])])
    result = compute_periodic_union_prefix_count(source, 0)
    assert result.count == 0


def test_periodicity() -> None:
    """Extending the cutoff by one period increases the count by occupied_count."""
    source = _source([("2", ["0"])])
    result_small = compute_periodic_union_prefix_count(source, 10)
    result_large = compute_periodic_union_prefix_count(source, 12)
    assert result_large.count - result_small.count == 1


def test_result_preserves_source() -> None:
    """Result retains the source and cutoff."""
    source = _source([("2", ["0"])])
    result = compute_periodic_union_prefix_count(source, 10)
    assert result.source == source
    assert result.cutoff == "10"
