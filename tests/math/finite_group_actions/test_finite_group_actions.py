"""Known-answer and adversarial tests for finite group-action operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.finite_group_actions._models import (
    BurnsideCountRequest,
    CycleIndexRequest,
    ElementCyclesRequest,
    FinitePermutationAction,
    PolyaInventoryRequest,
)
from jacobian.math.finite_group_actions._operations import (
    _enumerate_group,
    compute_burnside_count,
    compute_cycle_index,
    compute_element_cycles,
    compute_polya_inventory,
)


def _assert_error_type(error: ValidationError, expected: str) -> None:
    assert error.errors()[0]["type"] == expected


# ---------------------------------------------------------------------------
# Shared known actions
# ---------------------------------------------------------------------------


def _trivial(n: int) -> dict:
    """The trivial group acting on n labelled points."""
    return FinitePermutationAction(
        domain=tuple(f"p{i}" for i in range(n)),
        generators=(
            tuple(
                range(n),
            ),
        ),
    )


def _cyclic_c3() -> FinitePermutationAction:
    return FinitePermutationAction(
        domain=("a", "b", "c"),
        generators=((1, 2, 0),),
    )


def _dihedral_d4() -> FinitePermutationAction:
    """D_4 acting on the four vertices of a square."""
    # Rotation by 90 degrees and a reflection through a mid-edge axis.
    return FinitePermutationAction(
        domain=("v0", "v1", "v2", "v3"),
        generators=((1, 2, 3, 0), (1, 0, 3, 2)),
    )


def _symmetric_s3() -> FinitePermutationAction:
    return FinitePermutationAction(
        domain=("p0", "p1", "p2"),
        generators=((1, 2, 0), (1, 0, 2)),
    )


# ---------------------------------------------------------------------------
# Fixture 1: trivial action
# ---------------------------------------------------------------------------


class TestTrivialAction:
    def test_trivial_group_order(self) -> None:
        assert len(_enumerate_group(_trivial(4))) == 1

    def test_trivial_burnside_one_orbit(self) -> None:
        result = compute_burnside_count(BurnsideCountRequest(action=_trivial(5)))
        assert result.orbit_count == 5
        assert result.group_order == 1
        assert result.fixed_point_contributions == (5,)

    def test_trivial_cycle_index_identity(self) -> None:
        result = compute_cycle_index(CycleIndexRequest(action=_trivial(3)))
        assert result.cycle_type_counts == (((1, 1, 1), 1),)
        assert result.group_order == 1
        assert result.degree == 3


# ---------------------------------------------------------------------------
# Fixtures 2-4: cyclic, dihedral, symmetric actions
# ---------------------------------------------------------------------------


class TestStandardActions:
    def test_cyclic_c3_order(self) -> None:
        assert len(_enumerate_group(_cyclic_c3())) == 3

    def test_dihedral_d4_order(self) -> None:
        assert len(_enumerate_group(_dihedral_d4())) == 8

    def test_symmetric_s3_order(self) -> None:
        assert len(_enumerate_group(_symmetric_s3())) == 6

    def test_symmetric_s4_order(self) -> None:
        s4 = FinitePermutationAction(
            domain=("p0", "p1", "p2", "p3"),
            generators=((1, 2, 3, 0), (1, 0, 2, 3)),
        )
        assert len(_enumerate_group(s4)) == 24


# ---------------------------------------------------------------------------
# Fixture 5: action with several point orbits
# ---------------------------------------------------------------------------


class TestMultipleOrbits:
    def test_disjoint_cycles_give_multiple_orbits(self) -> None:
        # C_2 acting separately on {a,b} and fixing {c,d}.
        action = FinitePermutationAction(
            domain=("a", "b", "c", "d"),
            generators=((1, 0, 2, 3),),
        )
        result = compute_burnside_count(BurnsideCountRequest(action=action))
        assert result.orbit_count == 3


# ---------------------------------------------------------------------------
# Fixture 6: fixed points plus nontrivial cycles
# ---------------------------------------------------------------------------


class TestFixedPointsAndCycles:
    def test_transposition_has_fixed_points(self) -> None:
        # Element that swaps p0,p1 and fixes p2.
        action = _symmetric_s3()
        # Enumerate and find an element with exactly one fixed point.
        group = _enumerate_group(action)
        # A transposition (0,1,2) -> (1,0,2) should exist; it has 1 fixed point.
        target = (1, 0, 2)
        idx = list(group).index(target)
        result = compute_element_cycles(
            ElementCyclesRequest(action=action, element=idx)
        )
        assert result.fixed_point_count == 1
        assert result.fixed_points == (2,)
        assert result.cycle_type == (2, 1)

    def test_identity_all_fixed(self) -> None:
        action = _cyclic_c3()
        result = compute_element_cycles(ElementCyclesRequest(action=action, element=0))
        assert result.fixed_point_count == 3
        assert result.cycles == ((0,), (1,), (2,))
        assert result.cycle_type == (1, 1, 1)
        assert result.support == ()


# ---------------------------------------------------------------------------
# Fixture 7: cycle-type profile and group-order total
# ---------------------------------------------------------------------------


class TestCycleTypeProfile:
    def test_s3_cycle_type_counts_total_group_order(self) -> None:
        result = compute_cycle_index(CycleIndexRequest(action=_symmetric_s3()))
        total = sum(count for _, count in result.cycle_type_counts)
        assert total == result.group_order == 6
        counts_dict = dict(result.cycle_type_counts)
        assert counts_dict == {(1, 1, 1): 1, (2, 1): 3, (3,): 2}

    def test_cyclic_c3_cycle_type_counts(self) -> None:
        result = compute_cycle_index(CycleIndexRequest(action=_cyclic_c3()))
        counts_dict = dict(result.cycle_type_counts)
        assert counts_dict == {(1, 1, 1): 1, (3,): 2}


# ---------------------------------------------------------------------------
# Fixture 8: exact cycle-index coefficients
# ---------------------------------------------------------------------------


class TestCycleIndexCoefficients:
    def test_s3_cycle_index_is_z3_formula(self) -> None:
        # Z(S_3) = (1/6)(x1^3 + 3 x1 x2 + 2 x3).
        result = compute_cycle_index(CycleIndexRequest(action=_symmetric_s3()))
        counts = dict(result.cycle_type_counts)
        # (1,1,1): identity -> 1; (2,1): 3 transpositions; (3,): 2 three-cycles.
        assert counts[(1, 1, 1)] == 1
        assert counts[(2, 1)] == 3
        assert counts[(3,)] == 2

    def test_d4_cycle_index(self) -> None:
        # Z(D_4) = (1/8)(x1^4 + 2 x1^2 x2 + 3 x2^2 + 2 x4).
        result = compute_cycle_index(CycleIndexRequest(action=_dihedral_d4()))
        counts = dict(result.cycle_type_counts)
        assert counts[(1, 1, 1, 1)] == 1
        assert counts[(2, 1, 1)] == 2
        assert counts[(2, 2)] == 3
        assert counts[(4,)] == 2
        assert result.group_order == 8


# ---------------------------------------------------------------------------
# Fixture 9: Burnside point-orbit count agrees with direct orbit partition
# ---------------------------------------------------------------------------


class TestBurnsideOrbitPartition:
    def _direct_orbit_count(self, action: FinitePermutationAction) -> int:
        group = _enumerate_group(action)
        n = len(action.domain)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for perm in group:
            for i in range(n):
                a, b = find(i), find(perm[i])
                if a != b:
                    parent[max(a, b)] = min(a, b)
        return len({find(i) for i in range(n)})

    def test_s3_burnside_matches_partition(self) -> None:
        action = _symmetric_s3()
        result = compute_burnside_count(BurnsideCountRequest(action=action))
        assert result.orbit_count == self._direct_orbit_count(action)

    def test_d4_burnside_matches_partition(self) -> None:
        action = _dihedral_d4()
        result = compute_burnside_count(BurnsideCountRequest(action=action))
        assert result.orbit_count == self._direct_orbit_count(action)

    def test_multiple_orbit_action_burnside_matches_partition(self) -> None:
        action = FinitePermutationAction(
            domain=("a", "b", "c", "d", "e"),
            generators=((1, 0, 3, 2, 4),),
        )
        result = compute_burnside_count(BurnsideCountRequest(action=action))
        assert result.orbit_count == self._direct_orbit_count(action)


# ---------------------------------------------------------------------------
# Fixture 10: two-colour unrestricted colouring counts
# ---------------------------------------------------------------------------


class TestTwoColorCounts:
    def test_s3_two_color_total_orbits(self) -> None:
        # S_3 on 3 points with 2 colours: 4 orbits.
        result = compute_polya_inventory(
            PolyaInventoryRequest(action=_symmetric_s3(), colors=2)
        )
        assert sum(c for _, c in result.terms) == 4

    def test_s3_two_color_polynomial_coefficients(self) -> None:
        result = compute_polya_inventory(
            PolyaInventoryRequest(action=_symmetric_s3(), colors=2)
        )
        assert dict(result.terms) == {(0, 3): 1, (1, 2): 1, (2, 1): 1, (3, 0): 1}

    def test_trivial_two_color_is_binomial(self) -> None:
        # Trivial action on n points: every colouring is its own orbit.
        action = _trivial(4)
        result = compute_polya_inventory(PolyaInventoryRequest(action=action, colors=2))
        assert sum(c for _, c in result.terms) == 16


# ---------------------------------------------------------------------------
# Fixture 11: subset-orbit inventory by cardinality
# ---------------------------------------------------------------------------


class TestSubsetInventory:
    def test_s3_subset_inventory_by_cardinality(self) -> None:
        # For 2 colours (absent/present), the coefficient of t^k counts the
        # number of S_3 orbits on k-subsets of a 3-set.
        # k=0: 1 (empty set), k=1: 1, k=2: 1, k=3: 1.
        result = compute_polya_inventory(
            PolyaInventoryRequest(action=_symmetric_s3(), colors=2)
        )
        by_degree = {mono[1]: coeff for mono, coeff in result.terms}
        assert by_degree == {0: 1, 1: 1, 2: 1, 3: 1}

    def test_d4_subset_inventory_by_cardinality(self) -> None:
        # D_4 on 4 vertices; orbits on subsets of each cardinality.
        result = compute_polya_inventory(
            PolyaInventoryRequest(action=_dihedral_d4(), colors=2)
        )
        by_degree = {mono[1]: coeff for mono, coeff in result.terms}
        # k=0: 1; k=1: 1; k=2: 2; k=3: 1; k=4: 1.
        assert by_degree == {0: 1, 1: 1, 2: 2, 3: 1, 4: 1}


# ---------------------------------------------------------------------------
# Fixture 12: labelled multicolour Pólya polynomial
# ---------------------------------------------------------------------------


class TestMulticolorPolya:
    def test_s3_three_color_total(self) -> None:
        result = compute_polya_inventory(
            PolyaInventoryRequest(action=_symmetric_s3(), colors=3)
        )
        assert sum(c for _, c in result.terms) == 10

    def test_s3_three_color_monomials_sum_to_degree(self) -> None:
        result = compute_polya_inventory(
            PolyaInventoryRequest(action=_symmetric_s3(), colors=3)
        )
        for mono, _ in result.terms:
            assert sum(mono) == 3
            assert len(mono) == 3


# ---------------------------------------------------------------------------
# Fixture 13: tuple orbit count for several lengths
# ---------------------------------------------------------------------------


class TestTupleOrbitCount:
    def test_s3_tuple_orbit_via_burnside_on_diagonal(self) -> None:
        # For the diagonal action on X^r, |Fix_{X^r}(g)| = |Fix_X(g)|^r.
        # The orbit count equals (1/|G|) sum_g |Fix_X(g)|^r, which we can
        # verify by expanding the tuple action explicitly for small r.
        action = _symmetric_s3()
        # Direct tuple Burnside using fixed-point powers.
        group = _enumerate_group(action)
        for r in (1, 2, 3):
            total = sum(_fixed_points(perm) ** r for perm in group)
            orbit_count = total // len(group)
            # For r=1 this should match the usual Burnside count.
            if r == 1:
                result = compute_burnside_count(BurnsideCountRequest(action=action))
                assert orbit_count == result.orbit_count

            # Verify by brute-force orbit enumeration on X^r.
            brute = _brute_tuple_orbit_count(action, r)
            assert orbit_count == brute

    def test_tuple_orbit_count_grows_with_length(self) -> None:
        action = _cyclic_c3()
        counts = []
        for r in range(1, 4):
            group = _enumerate_group(action)
            total = sum(_fixed_points(perm) ** r for perm in group)
            counts.append(total // len(group))
        # Orbits should be non-decreasing in r.
        assert counts == sorted(counts)


def _fixed_points(perm: tuple[int, ...]) -> int:
    return sum(1 for i in range(len(perm)) if perm[i] == i)


def _brute_tuple_orbit_count(action: FinitePermutationAction, r: int) -> int:
    n = len(action.domain)
    group = _enumerate_group(action)
    all_tuples = tuple(_itertools_product(range(n), repeat=r))

    parent: dict[tuple[int, ...], tuple[int, ...]] = {t: t for t in all_tuples}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for perm in group:
        for t in all_tuples:
            image = tuple(perm[t[i]] for i in range(r))
            union(t, image)
    return len({find(t) for t in all_tuples})


def _itertools_product(*args, **kwargs):
    import itertools

    return itertools.product(*args, **kwargs)


# ---------------------------------------------------------------------------
# Fixture 14: action conjugation and domain relabelling covariance
# ---------------------------------------------------------------------------


class TestConjugationCovariance:
    def test_relabelling_preserves_cycle_index(self) -> None:
        """Relabelling the domain should not change cycle-type counts."""
        action = _symmetric_s3()
        relabeled = FinitePermutationAction(
            domain=("alpha", "beta", "gamma"),
            generators=((1, 2, 0), (1, 0, 2)),
        )
        r1 = compute_cycle_index(CycleIndexRequest(action=action))
        r2 = compute_cycle_index(CycleIndexRequest(action=relabeled))
        assert r1.cycle_type_counts == r2.cycle_type_counts
        assert r1.group_order == r2.group_order

    def test_conjugation_preserves_burnside_count(self) -> None:
        """A conjugate action has the same Burnside orbit count."""
        action = _cyclic_c3()
        # Conjugate by the permutation (0->2->1->0): swap labels b,c.
        # This relabelling permutes the generators accordingly.
        conjugate = FinitePermutationAction(
            domain=("a", "c", "b"),
            generators=((2, 0, 1),),
        )
        r1 = compute_burnside_count(BurnsideCountRequest(action=action))
        r2 = compute_burnside_count(BurnsideCountRequest(action=conjugate))
        assert r1.orbit_count == r2.orbit_count


# ---------------------------------------------------------------------------
# Fixture 15: generator-order invariance
# ---------------------------------------------------------------------------


class TestGeneratorOrderInvariance:
    def test_generator_reorder_preserves_group(self) -> None:
        g1 = FinitePermutationAction(
            domain=("a", "b", "c"),
            generators=((1, 2, 0), (1, 0, 2)),
        )
        g2 = FinitePermutationAction(
            domain=("a", "b", "c"),
            generators=((1, 0, 2), (1, 2, 0)),
        )
        assert set(_enumerate_group(g1)) == set(_enumerate_group(g2))

    def test_reorder_preserves_cycle_index(self) -> None:
        g1 = FinitePermutationAction(
            domain=("a", "b", "c"),
            generators=((1, 2, 0), (1, 0, 2)),
        )
        g2 = FinitePermutationAction(
            domain=("a", "b", "c"),
            generators=((1, 0, 2), (1, 2, 0)),
        )
        r1 = compute_cycle_index(CycleIndexRequest(action=g1))
        r2 = compute_cycle_index(CycleIndexRequest(action=g2))
        assert r1.cycle_type_counts == r2.cycle_type_counts


# ---------------------------------------------------------------------------
# Fixture 16: calls exactly at and immediately above bounds
# ---------------------------------------------------------------------------


class TestBounds:
    def test_duplicate_labels_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FinitePermutationAction(
                domain=("a", "b", "a"),
                generators=((1, 2, 0),),
            )
        _assert_error_type(
            exc_info.value, "finite_group_action.domain_labels_not_distinct"
        )

    def test_non_permutation_generator_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FinitePermutationAction(
                domain=("a", "b", "c"),
                generators=((1, 2, 1),),
            )
        _assert_error_type(
            exc_info.value, "finite_group_action.generator_not_permutation"
        )

    def test_wrong_length_generator_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            FinitePermutationAction(
                domain=("a", "b", "c"),
                generators=((1, 2),),
            )
        _assert_error_type(
            exc_info.value, "finite_group_action.generator_length_mismatch"
        )

    def test_element_index_out_of_range_rejected(self) -> None:
        action = _cyclic_c3()
        with pytest.raises(ValueError, match="out of range"):
            compute_element_cycles(ElementCyclesRequest(action=action, element=3))

    def test_colors_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PolyaInventoryRequest(action=_cyclic_c3(), colors=0)

    def test_group_order_bound_exceeded(self) -> None:
        # S_8 has order 40320 > 10000 and acts on only 8 points, so it fits
        # the domain-size cap but exceeds the group-order bound.
        action = FinitePermutationAction(
            domain=tuple(f"p{i}" for i in range(8)),
            generators=((1, 2, 3, 4, 5, 6, 7, 0), (1, 0, 2, 3, 4, 5, 6, 7)),
        )
        with pytest.raises(ValueError, match="exceeds the bounded maximum"):
            compute_cycle_index(CycleIndexRequest(action=action))


# ---------------------------------------------------------------------------
# Additional fail-closed binding tests
# ---------------------------------------------------------------------------


class TestFailClosedBinding:
    def test_wrong_orbit_count_raises(self) -> None:
        """Constructing a result with a wrong value should fail validation."""
        from jacobian.math.finite_group_actions._models import BurnsideCountResult

        action = _cyclic_c3()
        with pytest.raises(ValueError, match="orbit_count"):
            BurnsideCountResult(
                action=action,
                group_order=3,
                fixed_point_sum=3,
                orbit_count=99,
                fixed_point_contributions=(3, 0, 0),
            )

    def test_wrong_cycle_type_raises(self) -> None:
        from jacobian.math.finite_group_actions._models import CycleIndexResult

        action = _cyclic_c3()
        with pytest.raises(ValueError, match="cycle_type_counts"):
            CycleIndexResult(
                action=action,
                group_order=3,
                degree=3,
                cycle_type_counts=(((1, 1, 1), 1),),
            )
