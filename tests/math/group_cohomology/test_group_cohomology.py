"""Tests for group cohomology operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.group_cohomology._models import (
    GroupCohomologyRequest,
    PermutationGroup,
)
from jacobian.math.group_cohomology._operations import compute_group_cohomology


class TestGroupCohomology:
    """Test group cohomology computation."""

    def test_h0_is_field(self):
        """H^0(G, K) = K always (trivial action)."""
        req = GroupCohomologyRequest(
            group=PermutationGroup(degree=2, generators=((1, 0),)),
            prime=2,
            max_degree=2,
        )
        result = compute_group_cohomology(req)
        assert result.groups[0].betti == 1

    def test_group_order(self):
        """The result should report the group order."""
        req = GroupCohomologyRequest(
            group=PermutationGroup(degree=3, generators=((1, 0, 2), (0, 2, 1))),
            prime=3,
            max_degree=1,
        )
        result = compute_group_cohomology(req)
        assert result.group_order == 6

    def test_cochain_dimensions(self):
        """C^n has dimension |G|^n."""
        req = GroupCohomologyRequest(
            group=PermutationGroup(degree=2, generators=((1, 0),)),
            prime=5,
            max_degree=3,
        )
        result = compute_group_cohomology(req)
        assert result.groups[0].dimension == 1
        assert result.groups[1].dimension == 2
        assert result.groups[2].dimension == 4
        assert result.groups[3].dimension == 8

    def test_prime_reported(self):
        """The result should report the prime."""
        req = GroupCohomologyRequest(
            group=PermutationGroup(degree=2, generators=((1, 0),)),
            prime=7,
            max_degree=1,
        )
        result = compute_group_cohomology(req)
        assert result.prime == 7

    def test_trivial_group(self):
        """The trivial group has H^0 = K and higher groups = 0."""
        req = GroupCohomologyRequest(
            group=PermutationGroup(degree=1, generators=((0,),)),
            prime=2,
            max_degree=2,
        )
        result = compute_group_cohomology(req)
        assert result.groups[0].betti == 1
        assert result.group_order == 1


class TestExactBarComplex:
    """The kernel materializes the inhomogeneous bar complex exactly."""

    def _compute(self, degree, generators, prime, max_degree=3):
        request = GroupCohomologyRequest(
            group=PermutationGroup(degree=degree, generators=generators),
            prime=prime,
            max_degree=max_degree,
        )
        return compute_group_cohomology(request)

    def test_c2_over_gf2_has_betti_one(self):
        result = self._compute(2, ((1, 0),), 2)
        bettis = {g.degree: g.betti for g in result.groups}
        assert bettis[1] == 1

    def test_cyclic_p_modular_series(self):
        """H*(C_p; GF(p)) has betti 1 in every positive degree."""
        result = self._compute(3, ((1, 2, 0),), 3)
        bettis = {g.degree: g.betti for g in result.groups}
        assert bettis == {0: 1, 1: 1, 2: 1, 3: 1}

    def test_trivial_group_higher_homology_vanishes(self):
        result = self._compute(1, ((0,),), 5)
        bettis = {g.degree: g.betti for g in result.groups}
        assert bettis == {0: 1, 1: 0, 2: 0, 3: 0}

    def test_coprime_characteristic_vanishes(self):
        """p not dividing |G| kills all higher cohomology."""
        result = self._compute(2, ((1, 0),), 3)
        bettis = {g.degree: g.betti for g in result.groups}
        assert bettis == {0: 1, 1: 0, 2: 0, 3: 0}

    def test_composite_prime_rejected_at_model(self):
        with pytest.raises(ValidationError, match="prime"):
            GroupCohomologyRequest(
                group=PermutationGroup(degree=2, generators=((1, 0),)),
                prime=4,
                max_degree=2,
            )

    def test_oversized_enumerated_order_rejected(self):
        with pytest.raises(ValidationError, match="bounded maximum"):
            PermutationGroup(
                degree=6,
                generators=((1, 0, 2, 3, 4, 5), (1, 2, 3, 4, 5, 0)),
            )

    def test_cochain_budget_rejected(self):
        """C4 x max_degree 4 would need 4^5 cochain elements; cap admits it,
        so force rejection through order instead."""
        with pytest.raises(ValidationError, match="budget"):
            GroupCohomologyRequest(
                group=PermutationGroup(degree=6, generators=((*tuple(range(1, 6)), 0),)),
                prime=2,
                max_degree=4,
            )
