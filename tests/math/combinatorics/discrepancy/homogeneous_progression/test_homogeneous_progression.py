from __future__ import annotations

from jacobian.math.combinatorics.discrepancy.homogeneous_progression.operations import (
    construct_homogeneous_progression_set_system,
)


def test_n0() -> None:
    """Empty ground set has no progressions."""
    result = construct_homogeneous_progression_set_system(0)
    assert result.ground_set_size == 0
    assert len(result.sets) == 0


def test_n1() -> None:
    """n=1: one progression {0} (representing {1})."""
    result = construct_homogeneous_progression_set_system(1)
    assert result.ground_set_size == 1
    assert (0,) in result.sets


def test_n4_fixture() -> None:
    """n=4: progressions include {0,1,2,3} for d=1 and {1,3} for d=2."""
    result = construct_homogeneous_progression_set_system(4)
    sets = [set(s) for s in result.sets]
    assert {0, 1, 2, 3} in sets  # d=1, k=4: {1,2,3,4} -> {0,1,2,3}
    assert {1, 3} in sets  # d=2, k=2: {2,4} -> {1,3}


def test_zero_based_indexing() -> None:
    """All indices are 0-based (representing 1..n)."""
    result = construct_homogeneous_progression_set_system(6)
    for s in result.sets:
        for idx in s:
            assert 0 <= idx < 6


def test_replay_progressions() -> None:
    """Replay: each row is d * initial_interval with zero-based translation."""
    n = 8
    result = construct_homogeneous_progression_set_system(n)
    for s in result.sets:
        original = tuple(idx + 1 for idx in s)
        d = original[0]
        for i, val in enumerate(original):
            assert val == d * (i + 1)


def test_count_identity() -> None:
    """Number of sets = sum_{d=1}^{n} floor(n/d)."""
    for n in [0, 1, 2, 4, 6, 10]:
        result = construct_homogeneous_progression_set_system(n)
        expected = sum(n // d for d in range(1, n + 1))
        assert len(result.sets) == expected, (
            f"n={n}: got {len(result.sets)}, expected {expected}"
        )


def test_no_duplicates() -> None:
    """Every progression appears exactly once."""
    result = construct_homogeneous_progression_set_system(10)
    seen = set()
    for s in result.sets:
        key = tuple(s)
        assert key not in seen
        seen.add(key)


def test_exhaustive_dk_enumeration() -> None:
    """Check every valid (d,k) pair is present."""
    n = 6
    result = construct_homogeneous_progression_set_system(n)
    expected_sets = set()
    for d in range(1, n + 1):
        k = 1
        while d * k <= n:
            progression = tuple(d * i - 1 for i in range(1, k + 1))
            expected_sets.add(progression)
            k += 1
    actual_sets = {tuple(s) for s in result.sets}
    assert actual_sets == expected_sets


def test_result_preserves_n() -> None:
    """Result retains the source n."""
    result = construct_homogeneous_progression_set_system(5)
    assert result.ground_set_size == 5


def test_native_negative_n_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="between 0"):
        construct_homogeneous_progression_set_system(-1)
