"""Tests for finite group invariants: conjugacy classes and subgroup lattice."""

from __future__ import annotations

import pytest

from jacobian.math.group._models import (
    GroupConjugacyClassesRequest,
    GroupSubgroupLatticeRequest,
)
from jacobian.math.group._operations import (
    compute_conjugacy_classes,
    compute_subgroup_lattice,
)


class TestConjugacyClasses:
    """Tests for ``group.conjugacy_classes.compute``."""

    def test_s3_conjugacy_classes(self):
        """S3 has 3 conjugacy classes."""
        req = GroupConjugacyClassesRequest(
            degree=3,
            generators=((1, 0, 2), (0, 2, 1)),
        )
        result = compute_conjugacy_classes(req)
        assert result.class_count == 3

    def test_cyclic_group_conjugacy_classes(self):
        """C4 (cyclic group of order 4) has 4 conjugacy classes (abelian)."""
        req = GroupConjugacyClassesRequest(
            degree=4,
            generators=((1, 2, 3, 0),),
        )
        result = compute_conjugacy_classes(req)
        assert result.class_count == 4

    def test_trivial_group(self):
        """The trivial group has 1 conjugacy class."""
        req = GroupConjugacyClassesRequest(
            degree=1,
            generators=((0,),),
        )
        result = compute_conjugacy_classes(req)
        assert result.class_count == 1
        assert result.classes[0].size == 1

    def test_class_sizes_sum_to_order(self):
        """The sum of class sizes equals the group order."""
        # D4 as permutation group on 4 points: <(0 1 2 3), (1 3)>
        req = GroupConjugacyClassesRequest(
            degree=4,
            generators=((1, 0, 3, 2), (0, 2, 1, 3)),
        )
        result = compute_conjugacy_classes(req)
        total = sum(c.size for c in result.classes)
        assert total == 8  # D4 has order 8


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
        """S6 (order 720) is rejected by the request model, not mid-run."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="bounded maximum"):
            GroupSubgroupLatticeRequest(
                degree=6,
                generators=((1, 0, 2, 3, 4, 5), (1, 2, 3, 4, 5, 0)),
            )
        with pytest.raises(ValidationError, match="bounded maximum"):
            GroupConjugacyClassesRequest(
                degree=6,
                generators=((1, 0, 2, 3, 4, 5), (1, 2, 3, 4, 5, 0)),
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

    def test_conjugacy_classes_serialize_all_elements(self):
        """Class sizes sum to the bounded group order."""
        req = GroupConjugacyClassesRequest(
            degree=3,
            generators=((1, 0, 2), (0, 2, 1)),
        )
        result = compute_conjugacy_classes(req)
        assert sum(cls.size for cls in result.classes) == 6
