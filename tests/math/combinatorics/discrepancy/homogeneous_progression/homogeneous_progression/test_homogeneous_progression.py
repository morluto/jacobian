from __future__ import annotations

from jacobian.math.combinatorics.discrepancy.homogeneous_progression.operations import (
    construct_homogeneous_progression_set_system,
)


def test_n_0() -> None:
    result = construct_homogeneous_progression_set_system(0)
    assert result.set_system.ground_set_size == 0
    assert len(result.set_system.sets) == 0


def test_n_1() -> None:
    result = construct_homogeneous_progression_set_system(1)
    assert result.set_system.ground_set_size == 1
    # Only set: (1,) -> 0-based: (0,)
    assert (0,) in result.set_system.sets


def test_n_6() -> None:
    result = construct_homogeneous_progression_set_system(6)
    ss = result.set_system
    # d=1: (0,), (0,1), (0,1,2), (0,1,2,3), (0,1,2,3,4), (0,1,2,3,4,5) -> 6 sets
    # d=2: (1,), (1,3), (1,3,5) -> 3 sets
    # d=3: (2,), (2,5) -> 2 sets
    # d=4: (3,) -> 1 set
    # d=5: (4,) -> 1 set
    # d=6: (5,) -> 1 set
    assert len(ss.sets) == 6 + 3 + 2 + 1 + 1 + 1


def test_all_sets_are_progressions() -> None:
    result = construct_homogeneous_progression_set_system(10)
    for subset in result.set_system.sets:
        if len(subset) == 1:
            continue
        diffs = [subset[i + 1] - subset[i] for i in range(len(subset) - 1)]
        assert len(set(diffs)) == 1, f"Non-constant difference in {subset}"
        assert all(0 <= x < 10 for x in subset)


def test_zero_based() -> None:
    result = construct_homogeneous_progression_set_system(4)
    # d=2, k=2 -> values 2,4 -> 0-based: 1,3
    assert (1, 3) in result.set_system.sets


def test_result_preserves_n() -> None:
    result = construct_homogeneous_progression_set_system(5)
    assert result.n == 5
    assert result.set_system.ground_set_size == 5
