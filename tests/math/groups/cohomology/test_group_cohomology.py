"""Tests for group cohomology operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.groups._models import PermutationGroup
from jacobian.math.groups.cohomology._models import (
    MAX_COCHAIN_DEGREE,
    CohomologyGroup,
    GroupCohomologyRequest,
    GroupCohomologyResult,
)
from jacobian.math.groups.cohomology._operations import compute_group_cohomology


class TestGroupCohomology:
    """Test group cohomology computation."""

    def test_h0_is_field(self) -> None:
        """H^0(G, K) = K always (trivial action)."""
        req = GroupCohomologyRequest(
            group=PermutationGroup(degree=2, generators=((1, 0),)),
            prime=2,
            max_degree=2,
        )
        result = compute_group_cohomology(req)
        assert result.groups[0].betti == 1

    def test_group_order(self) -> None:
        """The result should report the group order."""
        req = GroupCohomologyRequest(
            group=PermutationGroup(degree=3, generators=((1, 0, 2), (0, 2, 1))),
            prime=3,
            max_degree=1,
        )
        result = compute_group_cohomology(req)
        assert result.group_order == 6

    def test_cochain_dimensions(self) -> None:
        """C^n has dimension |G|^n."""
        req = GroupCohomologyRequest(
            group=PermutationGroup(degree=2, generators=((1, 0),)),
            prime=5,
            max_degree=3,
        )
        result = compute_group_cohomology(req)
        assert result.groups[0].cochain_dimension == 1
        assert result.groups[1].cochain_dimension == 2
        assert result.groups[2].cochain_dimension == 4
        assert result.groups[3].cochain_dimension == 8

    def test_prime_reported(self) -> None:
        """The result should report the prime."""
        req = GroupCohomologyRequest(
            group=PermutationGroup(degree=2, generators=((1, 0),)),
            prime=7,
            max_degree=1,
        )
        result = compute_group_cohomology(req)
        assert result.request.prime == 7

    def test_trivial_group(self) -> None:
        """The trivial group has H^0 = K and higher groups = 0."""
        req = GroupCohomologyRequest(
            group=PermutationGroup(degree=1, generators=((0,),)),
            prime=2,
            max_degree=2,
        )
        result = compute_group_cohomology(req)
        assert result.groups[0].betti == 1
        assert result.group_order == 1

    def test_trivial_group_high_degree_admitted_and_exact(self) -> None:
        """Regression: degree admission derives from the coupled work
        budgets, not a fixed ceiling. For the trivial group every cochain
        space and coboundary is one-dimensional and both budgets stay at
        one, so a cheap exact higher-degree query must be admitted."""
        request = GroupCohomologyRequest(
            group=PermutationGroup(degree=1, generators=((0,),)),
            prime=2,
            max_degree=24,
        )
        result = compute_group_cohomology(request)
        assert result.group_order == 1
        assert [g.degree for g in result.groups] == list(range(25))
        assert result.groups[0].betti == 1
        assert all(g.betti == 0 for g in result.groups[1:])
        assert all(g.cochain_dimension == 1 for g in result.groups)

    def test_trivial_group_fallback_ceiling_is_the_only_remaining_cap(self) -> None:
        """The conservative fallback binds only order-1 requests; its
        boundary degree is admitted and one past it is rejected."""
        request = GroupCohomologyRequest(
            group=PermutationGroup(degree=1, generators=((0,),)),
            prime=3,
            max_degree=MAX_COCHAIN_DEGREE,
        )
        result = compute_group_cohomology(request)
        assert len(result.groups) == MAX_COCHAIN_DEGREE + 1
        assert result.groups[0].betti == 1
        with pytest.raises(ValidationError):
            GroupCohomologyRequest(
                group=PermutationGroup(degree=1, generators=((0,),)),
                prime=3,
                max_degree=MAX_COCHAIN_DEGREE + 1,
            )


class TestExactBarComplex:
    """The kernel materializes the inhomogeneous bar complex exactly."""

    def _compute(
        self,
        degree: int,
        generators: tuple[tuple[int, ...], ...],
        prime: int,
        max_degree: int = 3,
    ) -> GroupCohomologyResult:
        request = GroupCohomologyRequest(
            group=PermutationGroup(degree=degree, generators=generators),
            prime=prime,
            max_degree=max_degree,
        )
        return compute_group_cohomology(request)

    def test_c2_over_gf2_has_betti_one(self) -> None:
        result = self._compute(2, ((1, 0),), 2)
        bettis = {g.degree: g.betti for g in result.groups}
        assert bettis[1] == 1

    def test_cyclic_p_modular_series(self) -> None:
        """H*(C_p; GF(p)) has betti 1 in every positive degree."""
        result = self._compute(3, ((1, 2, 0),), 3)
        bettis = {g.degree: g.betti for g in result.groups}
        assert bettis == {0: 1, 1: 1, 2: 1, 3: 1}

    def test_trivial_group_higher_homology_vanishes(self) -> None:
        result = self._compute(1, ((0,),), 5)
        bettis = {g.degree: g.betti for g in result.groups}
        assert bettis == {0: 1, 1: 0, 2: 0, 3: 0}

    def test_coprime_characteristic_vanishes(self) -> None:
        """p not dividing |G| kills all higher cohomology."""
        result = self._compute(2, ((1, 0),), 3)
        bettis = {g.degree: g.betti for g in result.groups}
        assert bettis == {0: 1, 1: 0, 2: 0, 3: 0}

    def test_cochain_dimension_is_not_the_cohomology_dimension(self) -> None:
        """H^1(C2; GF(3)) is zero-dimensional inside a 2-dim cochain space."""
        result = self._compute(2, ((1, 0),), 3)
        first = result.groups[1]
        assert first.betti == 0
        assert first.cochain_dimension == 2

    def test_composite_prime_rejected_by_operation(self) -> None:
        with pytest.raises(ValueError, match="prime must be a prime integer"):
            compute_group_cohomology(
                GroupCohomologyRequest(
                    group=PermutationGroup(degree=2, generators=((1, 0),)),
                    prime=4,
                    max_degree=2,
                )
            )

    def test_oversized_enumerated_order_rejected(self) -> None:
        # The canonical permutation-group value allows degree up to 64 and
        # does not bound order itself; the cohomology outer request owns the
        # 64-element order budget.
        with pytest.raises(ValueError, match="exceeds the bounded maximum"):
            compute_group_cohomology(
                GroupCohomologyRequest(
                    group=PermutationGroup(
                        degree=6,
                        generators=((1, 0, 2, 3, 4, 5), (1, 2, 3, 4, 5, 0)),
                    ),
                    prime=2,
                    max_degree=1,
                )
            )

    def test_degree_above_sixteen_with_bounded_order_admitted(self) -> None:
        """The duplicate degree-16 cap is gone: a degree-20 action whose
        enumerated order stays bounded is admitted."""
        GroupCohomologyRequest(
            group=PermutationGroup(degree=20, generators=(tuple(range(20)),)),
            prime=2,
            max_degree=0,
        )

    def test_cochain_budget_rejected(self) -> None:
        """Order 6 at max_degree 4 would need 6^5 cochain elements; the
        work-derived degree budget for order 6 is 2 and rejects it."""
        with pytest.raises(ValueError, match="exceeds the work-derived degree budget"):
            compute_group_cohomology(
                GroupCohomologyRequest(
                    group=PermutationGroup(
                        degree=6, generators=((*tuple(range(1, 6)), 0),)
                    ),
                    prime=2,
                    max_degree=4,
                )
            )

    def test_dense_bar_matrix_budget_rejected(self) -> None:
        """Order 4 at max_degree 5 fits no derived envelope: its degree-5
        coboundary is a dense 4096x1024 matrix and the cell bound caps
        order 4 at degree 3."""
        with pytest.raises(ValueError, match="exceeds the work-derived degree budget"):
            compute_group_cohomology(
                GroupCohomologyRequest(
                    group=PermutationGroup(degree=4, generators=((1, 2, 3, 0),)),
                    prime=2,
                    max_degree=5,
                )
            )

    def test_dense_bar_matrix_budget_admits_bounded_requests(self) -> None:
        """C2 at the maximum degree and C4 at degree 3 stay inside the cells."""
        GroupCohomologyRequest(
            group=PermutationGroup(degree=2, generators=((1, 0),)),
            prime=2,
            max_degree=6,
        )
        GroupCohomologyRequest(
            group=PermutationGroup(degree=4, generators=((1, 2, 3, 0),)),
            prime=3,
            max_degree=3,
        )

    def test_derived_budget_rejects_oversized_work_before_kernel(self) -> None:
        """Order 32 admits degree 1, but its degree-8 coboundary would be a
        dense 32^17-cell matrix; the work-derived budget rejects it during
        request validation, before any kernel expansion runs."""
        with pytest.raises(ValueError, match="exceeds the work-derived degree budget"):
            compute_group_cohomology(
                GroupCohomologyRequest(
                    group=PermutationGroup(degree=5, generators=((1, 2, 3, 4, 0),)),
                    prime=2,
                    max_degree=8,
                )
            )

    def test_derived_budget_boundary_admitted(self) -> None:
        """Order 32 at its work-derived maximum degree 1 stays inside both
        budgets: 32^2 cochain elements and 32^3 matrix cells."""
        GroupCohomologyRequest(
            group=PermutationGroup(degree=5, generators=((1, 2, 3, 4, 0),)),
            prime=2,
            max_degree=1,
        )

    def test_c2_degree_seven_admitted_by_derived_budget(self) -> None:
        """Order 2's coupled budgets (2^15 cells <= 65536) admit degree 7,
        which the removed fixed ceiling rejected. Known answer:
        H*(C2; GF(2)) = GF(2)[x] has betti 1 in every degree."""
        request = GroupCohomologyRequest(
            group=PermutationGroup(degree=2, generators=((1, 0),)),
            prime=2,
            max_degree=7,
        )
        result = compute_group_cohomology(request)
        assert {g.degree: g.betti for g in result.groups} == dict.fromkeys(range(8), 1)

    def test_reuses_canonical_permutation_group_value(self) -> None:
        """GroupCohomologyRequest reuses PermutationGroup so native
        composition such as GroupCohomologyRequest(group=result.stabilizer)
        and result.request.group -> group consumer works unchanged."""
        from jacobian.math.groups._models import (
            PermutationGroup as CanonicalGroup,
        )
        from jacobian.math.groups.operations import group_order, group_stabilizer

        canonical = CanonicalGroup(degree=3, generators=((1, 0, 2), (0, 2, 1)))
        req = GroupCohomologyRequest(group=canonical, prime=2, max_degree=1)
        result = compute_group_cohomology(req)
        # result.request.group must be consumable by group consumers unchanged
        assert group_order(result.request.group) == 6

        # stabilizer result's stabilizer subgroup feeds cohomology unchanged
        source = CanonicalGroup(degree=4, generators=((1, 0, 2, 3), (1, 2, 3, 0)))
        stab = group_stabilizer(source, 0)
        GroupCohomologyRequest(group=stab, prime=2, max_degree=1)


class TestDeclarationContract:
    """The declaration must describe the implemented complex."""

    def test_description_names_unnormalized_inhomogeneous_complex(self) -> None:
        from jacobian.math.groups.cohomology._tools import TOOLS

        assert len(TOOLS) == 1
        description = TOOLS[0].description.lower()
        assert "unnormalized inhomogeneous bar complex" in description
        assert "the normalized bar complex" not in description

    def test_reported_dimensions_match_unnormalized_construction(self) -> None:
        """Unnormalized C^n = {functions G^n -> GF(p)} has dimension |G|^n."""
        request = GroupCohomologyRequest(
            group=PermutationGroup(degree=2, generators=((1, 0),)),
            prime=2,
            max_degree=2,
        )
        result = compute_group_cohomology(request)
        assert [g.cochain_dimension for g in result.groups] == [1, 2, 4]


class TestResultBinding:
    """Results are structural; explicit verification replays the bar complex."""

    def _request(self) -> GroupCohomologyRequest:
        return GroupCohomologyRequest(
            group=PermutationGroup(degree=3, generators=((1, 0, 2), (0, 2, 1))),
            prime=3,
            max_degree=2,
        )

    def test_result_retains_request_without_replaying(self) -> None:
        request = self._request()
        result = compute_group_cohomology(request)
        assert result.request == request
        assert GroupCohomologyResult(
            request=request,
            groups=result.groups,
            group_order=result.group_order,
        )

    def test_result_parsing_does_not_readmit_its_source_request(self) -> None:
        result = GroupCohomologyResult(
            request=GroupCohomologyRequest(
                group=PermutationGroup(degree=2, generators=((1, 0),)),
                prime=4,
                max_degree=1,
            ),
            groups=(
                CohomologyGroup(degree=0, betti=1, cochain_dimension=1),
                CohomologyGroup(degree=1, betti=0, cochain_dimension=2),
            ),
            group_order=2,
        )
        assert result.request.prime == 4
