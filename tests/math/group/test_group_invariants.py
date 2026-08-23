"""Tests for finite group invariants: subgroup lattice bounds and binding."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.group._models import (
    GroupConjugacyClassesRequest,
    GroupSubgroupLatticeRequest,
)
from jacobian.math.group._operations import (
    compute_subgroup_lattice,
)


class TestSubgroupLattice:
    """Test for ``group.subgroup_lattice.compute``."""

    def test_c4_subgroups(self):
        """C4 has 3 subgroups: trivial, C2, C4."""
        req = GroupSubgroupLatticeRequest(
            degree=4,
            generators=((1, 2, 3, 0),),
        )
        result = compute_subgroup_lattice(req)
        assert result.subgroup_count == 3

    def test_s3_subgroups(self):
        """S3 has 6 subgroups."""
        req = GroupSubgroupLatticeRequest(
            degree=3,
            generators=((1, 0, 2), (0, 2, 1)),
        )
        result = compute_subgroup_lattice(req)
        assert result.subgroup_count == 6

    def test_trivial_group_subgroups(self):
        """The trivial group has 1 subgroup (itself)."""
        req = GroupSubgroupLatticeRequest(
            degree=1,
            generators=((0,),),
        )
        result = compute_subgroup_lattice(req)
        assert result.subgroup_count == 1

    def test_subgroup_orders(self):
        """Subgroup orders should include 1 and the group order."""
        req = GroupSubgroupLatticeRequest(
            degree=4,
            generators=((1, 2, 3, 0),),
        )
        result = compute_subgroup_lattice(req)
        orders = sorted(sg.order for sg in result.subgroups)
        assert orders == [1, 2, 4]


class TestBoundedEnumeration:
    """Admission bounds the enumerated order; traversal stays tractable."""

    def test_oversized_group_rejected_at_validation(self):
        """Each operation rejects groups above its own enumerated-order cap:
        the lattice at 64, conjugacy classes at 5000."""
        s6_generators = ((1, 0, 2, 3, 4, 5), (1, 2, 3, 4, 5, 0))
        with pytest.raises(ValidationError, match="bounded maximum"):
            GroupSubgroupLatticeRequest(degree=6, generators=s6_generators)
        s7_generators = ((1, 0, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6, 0))
        with pytest.raises(ValidationError, match="bounded maximum"):
            GroupConjugacyClassesRequest(degree=7, generators=s7_generators)

    def test_lattice_still_rejects_above_64(self):
        with pytest.raises(ValidationError, match="bounded maximum"):
            GroupSubgroupLatticeRequest(
                degree=5,
                generators=((1, 0, 2, 3, 4), (1, 2, 3, 4, 0)),
            )

    def test_c32_lattice_is_divisor_chain(self):
        """C32 has exactly 6 subgroups, one per divisor of 32."""
        req = GroupSubgroupLatticeRequest(
            degree=32,
            generators=((*tuple(range(1, 32)), 0),),
        )
        result = compute_subgroup_lattice(req)
        assert result.subgroup_count == 6
        orders = sorted(sg.order for sg in result.subgroups)
        assert orders == [1, 2, 4, 8, 16, 32]


class TestLatticeResultBoundToSourceGroup:
    def _c4_result(self):
        request = GroupSubgroupLatticeRequest(
            degree=4,
            generators=((1, 2, 3, 0),),
        )
        return compute_subgroup_lattice(request)

    def test_result_retains_source_and_revalidates(self) -> None:
        from jacobian.math.group._models import GroupSubgroupLatticeResult

        result = self._c4_result()
        assert result.degree == 4
        payload = result.model_dump(mode="json")
        assert GroupSubgroupLatticeResult.model_validate(payload) == result

    def test_incomplete_lattice_rejected(self) -> None:
        """The degree-2 identity subgroup alone must not revalidate as C2's
        complete lattice, which also contains C2 itself."""
        from jacobian.math.group._models import GroupSubgroupLatticeResult

        payload = self._c4_result().model_dump(mode="json")
        trivial_only = next(
            entry for entry in payload["subgroups"] if entry["order"] == 1
        )
        payload["subgroups"] = [trivial_only]
        payload["subgroup_count"] = 1
        with pytest.raises(ValidationError, match="complete subgroup lattice"):
            GroupSubgroupLatticeResult.model_validate(payload)

    def test_foreign_group_entries_rejected(self) -> None:
        """Entries of one source group cannot be relayed under another."""
        from jacobian.math.group._models import GroupSubgroupLatticeResult

        s3_payload = compute_subgroup_lattice(
            GroupSubgroupLatticeRequest(degree=3, generators=((1, 0, 2),))
        ).model_dump(mode="json")
        s3_payload["degree"] = 4
        s3_payload["generators"] = [[1, 2, 3, 0]]
        with pytest.raises(ValidationError, match="complete subgroup lattice"):
            GroupSubgroupLatticeResult.model_validate(s3_payload)
