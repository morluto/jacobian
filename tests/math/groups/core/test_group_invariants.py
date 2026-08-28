"""Tests for finite group invariants: subgroup lattice bounds and binding."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups._models import (
    GroupConjugacyClassesRequest,
    GroupSubgroupLatticeRequest,
    GroupSubgroupLatticeResult,
    PermutationGroup,
)
from jacobian.math.groups._operations import (
    compute_subgroup_lattice,
)


@contextmanager
def _group_error(code: str) -> Iterator[None]:
    with pytest.raises(ValidationError) as info:
        yield
    assert info.value.errors()[0]["type"] == code


class TestSubgroupLattice:
    """Test for ``group.subgroup_lattice.compute``."""

    def test_c4_subgroups(self) -> None:
        """C4 has 3 subgroups: trivial, C2, C4."""
        req = GroupSubgroupLatticeRequest(
            degree=4,
            generators=((1, 2, 3, 0),),
        )
        result = compute_subgroup_lattice(req)
        assert result.subgroup_count == 3

    def test_s3_subgroups(self) -> None:
        """S3 has 6 subgroups."""
        req = GroupSubgroupLatticeRequest(
            degree=3,
            generators=((1, 0, 2), (0, 2, 1)),
        )
        result = compute_subgroup_lattice(req)
        assert result.subgroup_count == 6

    def test_trivial_group_subgroups(self) -> None:
        """The trivial group has 1 subgroup (itself)."""
        req = GroupSubgroupLatticeRequest(
            degree=1,
            generators=((0,),),
        )
        result = compute_subgroup_lattice(req)
        assert result.subgroup_count == 1

    def test_subgroup_orders(self) -> None:
        """Subgroup orders should include 1 and the group order."""
        req = GroupSubgroupLatticeRequest(
            degree=4,
            generators=((1, 2, 3, 0),),
        )
        result = compute_subgroup_lattice(req)
        assert result.subgroups is not None
        orders = sorted(sg.order for sg in result.subgroups)
        assert orders == [1, 2, 4]


class TestNativeSubgroupLattice:
    """The exported native lattice composes with canonical group values."""

    @staticmethod
    def _s3() -> PermutationGroup:
        return PermutationGroup(degree=3, generators=((1, 0, 2), (0, 2, 1)))

    def test_stabilizer_result_feeds_lattice_unchanged(self) -> None:
        """A ``group_stabilizer`` value passes to the lattice directly."""
        from jacobian.math.groups.operations import (
            group_order,
            group_stabilizer,
            subgroup_lattice,
        )

        stabilizer = group_stabilizer(self._s3(), 0)
        entries = subgroup_lattice(stabilizer)
        # The C2 stabilizer of point 0 in S3 has trivial + itself.
        assert sorted(entry.order for entry in entries) == [1, 2]
        for entry in entries:
            assert group_order(entry.group) == entry.order

    def test_native_lattice_matches_wire_entries(self) -> None:
        """The native result equals the wire result's subgroup entries."""
        result = compute_subgroup_lattice(
            GroupSubgroupLatticeRequest(degree=3, generators=((1, 0, 2), (0, 2, 1)))
        )
        assert result.subgroups is not None
        native = self._s3()
        from jacobian.math.groups.operations import subgroup_lattice

        assert list(subgroup_lattice(native)) == list(result.subgroups)

    def test_enumerated_subgroups_feed_group_order(self) -> None:
        """Every enumerated subgroup passes to ``group_order`` unchanged."""
        from jacobian.math.groups.operations import group_order, subgroup_lattice

        for entry in subgroup_lattice(self._s3()):
            assert group_order(entry.group) == entry.order

    def test_oversized_source_rejected_before_traversal(self) -> None:
        """The native function enforces its own enumerated-order bound."""
        from jacobian.math.groups.operations import subgroup_lattice

        s5 = PermutationGroup(degree=5, generators=((1, 0, 2, 3, 4), (1, 2, 3, 4, 0)))
        with pytest.raises(ValueError, match="bounded to groups of order"):
            subgroup_lattice(s5)


class TestBoundedEnumeration:
    """Admission bounds the enumerated order; traversal stays tractable."""

    def test_oversized_group_rejected_at_validation(self) -> None:
        """Each operation rejects groups above its own enumerated-order cap:
        the lattice at 64, conjugacy classes at 5000."""
        s6_generators = ((1, 0, 2, 3, 4, 5), (1, 2, 3, 4, 5, 0))
        with pytest.raises(OperationDomainValidationError, match="group order"):
            compute_subgroup_lattice(
                GroupSubgroupLatticeRequest(degree=6, generators=s6_generators)
            )
        s7_generators = ((1, 0, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6, 0))
        from jacobian.math.groups._operations import compute_group_conjugacy_classes

        with pytest.raises(OperationDomainValidationError, match="group order"):
            compute_group_conjugacy_classes(
                GroupConjugacyClassesRequest(degree=7, generators=s7_generators)
            )

    def test_lattice_still_rejects_above_64(self) -> None:
        with pytest.raises(OperationDomainValidationError, match="group order"):
            compute_subgroup_lattice(
                GroupSubgroupLatticeRequest(
                    degree=5,
                    generators=((1, 0, 2, 3, 4), (1, 2, 3, 4, 0)),
                )
            )

    def test_c32_lattice_is_divisor_chain(self) -> None:
        """C32 has exactly 6 subgroups, one per divisor of 32."""
        req = GroupSubgroupLatticeRequest(
            degree=32,
            generators=((*tuple(range(1, 32)), 0),),
        )
        result = compute_subgroup_lattice(req)
        assert result.subgroups is not None
        assert result.subgroup_count == 6
        orders = sorted(sg.order for sg in result.subgroups)
        assert orders == [1, 2, 4, 8, 16, 32]


class TestLatticeResultBoundToSourceGroup:
    def _c4_result(self) -> GroupSubgroupLatticeResult:
        request = GroupSubgroupLatticeRequest(
            degree=4,
            generators=((1, 2, 3, 0),),
        )
        return compute_subgroup_lattice(request)

    def test_result_retains_source_and_revalidates(self) -> None:
        from jacobian.math.groups._models import GroupSubgroupLatticeResult

        result = self._c4_result()
        assert result.degree == 4
        payload = result.model_dump(mode="json")
        assert GroupSubgroupLatticeResult.model_validate(payload) == result


class TestLatticeWorkBound:
    """Traversal work is bounded by search-node count, not only group order."""

    @staticmethod
    def _c2_power_six() -> GroupSubgroupLatticeRequest:
        # Six disjoint transpositions: the elementary abelian group C2^6 of
        # order 64, whose lattice is the extremal admitted one.
        generators = []
        for pair in range(6):
            form = list(range(12))
            form[2 * pair], form[2 * pair + 1] = form[2 * pair + 1], form[2 * pair]
            generators.append(tuple(form))
        return GroupSubgroupLatticeRequest(degree=12, generators=tuple(generators))

    @pytest.mark.scale
    def test_extremal_abelian_lattice_completes(self) -> None:
        """C2^6 has exactly 2825 subgroups (subspaces of F_2^6)."""
        result = compute_subgroup_lattice(self._c2_power_six())
        assert result.subgroups is not None
        assert result.outcome == "COMPUTED"
        assert result.subgroup_count == 2825
        orders = sorted(entry.order for entry in result.subgroups)
        assert orders[0] == 1 and orders[-1] == 64
        # Every entry's generator chain stays within log2(order) <= 6 links.
        assert max(len(entry.group.generators) for entry in result.subgroups) <= 6
        # Entries are canonical permutation-group values: chainable into
        # other permutation-group consumers unchanged.
        first = result.subgroups[0].group
        assert first.degree == 12 and len(first.generators) >= 1

    def test_budget_exhaustion_returns_typed_outcome(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exhausted closure budget yields LIMIT_EXCEEDED, not an error."""
        import jacobian.math.groups.operations as operations

        monkeypatch.setattr(operations, "MAX_SUBGROUP_LATTICE_CLOSURES", 3)
        result = compute_subgroup_lattice(
            GroupSubgroupLatticeRequest(degree=4, generators=((1, 2, 3, 0),))
        )
        assert result.outcome == "LIMIT_EXCEEDED"
        assert result.subgroups is None
        assert result.detail is not None and "closure constructions" in result.detail

    def test_exceeded_outcome_rejects_entries_and_missing_detail(self) -> None:
        from jacobian.math.groups._models import GroupSubgroupLatticeResult

        with _group_error("group.outcome_shape"):
            GroupSubgroupLatticeResult.model_validate(
                {
                    "outcome": "LIMIT_EXCEEDED",
                    "degree": 4,
                    "generators": ((1, 2, 3, 0),),
                    "detail": None,
                }
            )
        with _group_error("group.outcome_shape"):
            GroupSubgroupLatticeResult.model_validate(
                {
                    "outcome": "LIMIT_EXCEEDED",
                    "degree": 4,
                    "generators": ((1, 2, 3, 0),),
                    "subgroups": (),
                    "detail": "exceeded",
                }
            )


class TestRelayedPayloadBounds:
    """Forged relayed payloads are capped before nested backend work."""

    def test_entry_count_cap_before_nested_validation(self) -> None:
        """More than the extremal subgroup count is rejected pre-nesting."""
        from jacobian.math.groups._models import (
            MAX_SUBGROUP_LATTICE_ENTRIES,
            GroupSubgroupLatticeResult,
        )

        payload = compute_subgroup_lattice(
            GroupSubgroupLatticeRequest(degree=4, generators=((1, 2, 3, 0),))
        ).model_dump(mode="json")
        valid_entry = payload["subgroups"][0]
        payload["subgroups"] = [valid_entry] * (MAX_SUBGROUP_LATTICE_ENTRIES + 1)
        payload["subgroup_count"] = len(payload["subgroups"])
        with _group_error("group.lattice_entry_bound"):
            GroupSubgroupLatticeResult.model_validate(payload)

    def test_generator_count_cap_restored(self) -> None:
        """Subgroup values declare a schema-visible generator-count cap."""
        from jacobian.math.groups._models import SubgroupEntry

        with pytest.raises(ValidationError):
            SubgroupEntry.model_validate(
                {
                    "group": {
                        "degree": 2,
                        "generators": tuple([(0, 1)] * 65),
                    },
                    "order": 2,
                }
            )


class TestExceededOutcomeSourceBinding:
    """Relayed limit-exceeded payloads keep the source-binding invariants."""

    @staticmethod
    def _exceeded(
        degree: int = 4,
        generators: tuple[tuple[int, ...], ...] = ((1, 2, 3, 0),),
        **overrides: Any,
    ) -> GroupSubgroupLatticeResult:
        from jacobian.math.groups._models import GroupSubgroupLatticeResult

        payload = {
            "outcome": "LIMIT_EXCEEDED",
            "degree": degree,
            "generators": [list(g) for g in generators],
            "detail": "traversal exceeded budget",
            **overrides,
        }
        return GroupSubgroupLatticeResult.model_validate(payload)

    def test_fabricated_subgroup_count_rejected(self) -> None:
        with _group_error("group.outcome_shape"):
            self._exceeded(subgroup_count=5)

    def test_exceeded_payload_still_admits_only_real_sources(self) -> None:
        """A non-permutation generator cannot ride on an exceeded outcome."""
        with pytest.raises(ValidationError):
            self._exceeded(degree=2, generators=((999,),))

    def test_entries_chain_into_permutation_group_consumers(self) -> None:
        from jacobian.math.groups._models import PermutationGroup
        from jacobian.math.groups._operations import compute_group_order

        result = compute_subgroup_lattice(
            GroupSubgroupLatticeRequest(degree=4, generators=((1, 2, 3, 0),))
        )
        assert result.subgroups is not None
        order_two = next(e for e in result.subgroups if e.order == 2)
        # The canonical value feeds group.order.compute unchanged.
        assert (
            str(
                compute_group_order(
                    PermutationGroup.model_validate(order_two.group.model_dump())
                ).order
            )
            == "2"
        )
