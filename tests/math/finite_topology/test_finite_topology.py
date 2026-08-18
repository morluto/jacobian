"""Tests for finite topology operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.finite_topology.operations import (
    beat_points,
    closure,
    connected_components,
    interior,
    is_continuous,
    specialization_preorder,
)
from jacobian.math.finite_topology.values import FiniteTopology, PointMap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discrete_topology(n: int) -> FiniteTopology:
    """Discrete topology on n points: all subsets are open."""
    from itertools import combinations

    points = list(range(n))
    all_subsets: list[tuple[int, ...]] = [()]
    for size in range(1, n + 1):
        for combo in combinations(points, size):
            all_subsets.append(tuple(sorted(combo)))
    return FiniteTopology(
        point_count=n,
        open_sets=tuple(all_subsets),
    )


def _sierpinski_topology() -> FiniteTopology:
    return FiniteTopology(
        point_count=2,
        open_sets=((), (1,), (0, 1)),
    )


def _trivial_topology(n: int) -> FiniteTopology:
    return FiniteTopology(
        point_count=n,
        open_sets=((), tuple(range(n))),
    )


# ---------------------------------------------------------------------------
# Specialization preorder
# ---------------------------------------------------------------------------


class TestSpecializationPreorder:
    def test_discrete(self):
        topo = _discrete_topology(3)
        preorder = specialization_preorder(topo)
        # In discrete topology: preorder[i][j] iff i == j
        for i in range(3):
            for j in range(3):
                assert preorder[i][j] == (i == j)

    def test_sierpinski(self):
        topo = _sierpinski_topology()
        preorder = specialization_preorder(topo)
        # Open sets: {}, {1}, {0,1}
        # closure({0}) = {0} (0 is in all opens containing 0, which is just {0,1})
        # Actually: preorder[i][j] iff every open containing i also contains j
        # Open containing 0: {0,1} -> contains 0 and 1
        # Open containing 1: {1}, {0,1} -> contains 1
        # So preorder[0][0]=T, preorder[0][1]=T, preorder[1][0]=F, preorder[1][1]=T
        assert preorder[0] == (True, True)
        assert preorder[1] == (False, True)

    def test_trivial(self):
        topo = _trivial_topology(2)
        preorder = specialization_preorder(topo)
        # Only opens are {} and {0,1}
        # Every open containing 0 also contains 1 and vice versa
        assert preorder[0] == (True, True)
        assert preorder[1] == (True, True)


# ---------------------------------------------------------------------------
# Closure
# ---------------------------------------------------------------------------


class TestClosure:
    def test_discrete_closure(self):
        topo = _discrete_topology(3)
        result = closure(topo, (1,))
        assert result == frozenset({1})

    def test_sierpinski_closure(self):
        topo = _sierpinski_topology()
        # closure({1}) = complement of union of opens disjoint from {1}
        # Opens disjoint from {1}: {} -> empty union
        # So closure({1}) = {0,1} - {} = {0,1}
        # Wait: opens are {}, {1}, {0,1}
        # Opens disjoint from {1}: {} only (since {1} contains 1, {0,1} contains 1)
        # So closure({1}) = {0,1} - {} = {0,1}
        result = closure(topo, (1,))
        assert result == frozenset({0, 1})

    def test_sierpinski_closure_0(self):
        topo = _sierpinski_topology()
        # closure({0}): opens disjoint from {0}: {} and {1}
        # Union = {1}
        # closure = {0,1} - {1} = {0}
        result = closure(topo, (0,))
        assert result == frozenset({0})


# ---------------------------------------------------------------------------
# Interior
# ---------------------------------------------------------------------------


class TestInterior:
    def test_discrete_interior(self):
        topo = _discrete_topology(3)
        result = interior(topo, (0, 1))
        assert result == frozenset({0, 1})

    def test_sierpinski_interior(self):
        topo = _sierpinski_topology()
        # interior({0}) = largest open subset of {0} = {} (since {0} is not open)
        result = interior(topo, (0,))
        assert result == frozenset()


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------


class TestConnectedComponents:
    def test_discrete_connected(self):
        topo = _discrete_topology(3)
        result = connected_components(topo)
        assert len(result) == 3
        # Each point is its own component
        for comp in result:
            assert len(comp) == 1

    def test_trivial_connected(self):
        topo = _trivial_topology(3)
        result = connected_components(topo)
        assert len(result) == 1
        assert set(result[0]) == {0, 1, 2}

    def test_sierpinski_connected(self):
        topo = _sierpinski_topology()
        result = connected_components(topo)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Continuity
# ---------------------------------------------------------------------------


class TestIsContinuous:
    def test_identity_is_continuous(self):
        topo = _discrete_topology(2)
        identity = PointMap(
            domain_point_count=2,
            codomain_point_count=2,
            function=(0, 1),
        )
        assert is_continuous(topo, topo, identity)

    def test_constant_map_continuous(self):
        domain = _discrete_topology(3)
        codomain = _trivial_topology(2)
        constant = PointMap(
            domain_point_count=3,
            codomain_point_count=2,
            function=(0, 0, 0),
        )
        # Preimage of any open in codomain is either empty or full set
        # In discrete topology both are open
        assert is_continuous(domain, codomain, constant)

    def test_non_continuous(self):
        domain = _trivial_topology(2)
        codomain = _discrete_topology(2)
        identity = PointMap(
            domain_point_count=2,
            codomain_point_count=2,
            function=(0, 1),
        )
        # Preimage of {0} = {0}, which is not open in trivial topology
        assert not is_continuous(domain, codomain, identity)


# ---------------------------------------------------------------------------
# Beat points
# ---------------------------------------------------------------------------


class TestBeatPoints:
    def test_discrete_no_beat_points(self):
        topo = _discrete_topology(3)
        down, up = beat_points(topo)
        # In discrete topology: no beat points
        assert len(down) == 0
        assert len(up) == 0

    def test_trivial_has_beat_points(self):
        topo = _trivial_topology(2)
        down, up = beat_points(topo)
        assert 0 in down
        assert 0 in up

    def test_sierpinski_beat_points(self):
        topo = _sierpinski_topology()
        down, up = beat_points(topo)
        # In Sierpinski: preorder[0]=(T,T), preorder[1]=(F,T)
        # below 0 = {1}, maximals of {1} = {1} -> unique -> 0 is down beat
        # above 1 = {0}, minimals of {0} = {0} -> unique -> 1 is up beat
        assert 0 in down
        assert 1 in up


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_empty_set_rejected(self):
        with pytest.raises(ValidationError):
            FiniteTopology(
                point_count=2,
                open_sets=((1,), (0, 1)),
            )

    def test_missing_full_set_rejected(self):
        with pytest.raises(ValidationError):
            FiniteTopology(
                point_count=2,
                open_sets=((), (1,)),
            )

    def test_point_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            FiniteTopology(
                point_count=2,
                open_sets=((), (0, 5)),
            )

    def test_duplicate_open_set_rejected(self):
        with pytest.raises(ValidationError):
            FiniteTopology(
                point_count=2,
                open_sets=((), (0, 1), (0, 1)),
            )
