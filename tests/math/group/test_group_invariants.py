"""Tests for finite group invariants: conjugacy classes and subgroup lattice."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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
        """Each operation rejects groups above its own enumerated-order cap:
        the lattice at 64, conjugacy classes at 5000."""
        from pydantic import ValidationError

        s6_generators = ((1, 0, 2, 3, 4, 5), (1, 2, 3, 4, 5, 0))
        with pytest.raises(ValidationError, match="bounded maximum"):
            GroupSubgroupLatticeRequest(degree=6, generators=s6_generators)
        s7_generators = ((1, 0, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6, 0))
        with pytest.raises(ValidationError, match="bounded maximum"):
            GroupConjugacyClassesRequest(degree=7, generators=s7_generators)

    def test_s5_conjugacy_classes_keep_the_prior_capability(self):
        """S5 has order 120: always admissible for conjugacy classes, which
        serialize each element once, while only the subgroup-lattice
        traversal carries the tighter order-64 cap."""
        request = GroupConjugacyClassesRequest(
            degree=5,
            generators=((1, 0, 2, 3, 4), (1, 2, 3, 4, 0)),
        )
        result = compute_conjugacy_classes(request)
        assert result.class_count == 7
        assert sum(cls.size for cls in result.classes) == 120

    def test_lattice_still_rejects_above_64(self):
        from pydantic import ValidationError

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

    def test_conjugacy_classes_serialize_all_elements(self):
        """Class sizes sum to the bounded group order."""
        req = GroupConjugacyClassesRequest(
            degree=3,
            generators=((1, 0, 2), (0, 2, 1)),
        )
        result = compute_conjugacy_classes(req)
        assert sum(cls.size for cls in result.classes) == 6


class TestSourceBoundGroupResults:
    def test_conjugacy_class_size_matches_elements(self) -> None:
        from jacobian.math.group._models import ConjugacyClass

        with pytest.raises(ValueError, match="class size"):
            ConjugacyClass(elements=((1, 0, 2),), size=2)

    def test_subgroup_order_must_match_generators(self) -> None:
        from jacobian.math.group._models import SubgroupEntry

        with pytest.raises(ValueError, match="does not match the order"):
            SubgroupEntry(generators=((1, 0),), order=1)
        accepted = SubgroupEntry(generators=((1, 0),), order=2)
        assert accepted.order == 2

    def test_subgroup_order_capped_at_enforced_limit(self) -> None:
        from jacobian.math.group._models import SubgroupEntry

        with pytest.raises(ValueError):
            # A single degree-64 cycle has order 64; claim one more.
            SubgroupEntry(
                generators=((*range(1, 64), 0),),
                order=65,
            )


class TestConjugacyResultBoundToSourceGroup:
    def _s3_result(self):
        request = GroupConjugacyClassesRequest(
            degree=3,
            generators=((1, 0, 2), (0, 2, 1)),
        )
        return compute_conjugacy_classes(request)

    def test_result_retains_source_and_revalidates(self) -> None:
        from jacobian.math.group._models import GroupConjugacyClassesResult

        result = self._s3_result()
        assert result.degree == 3
        payload = result.model_dump(mode="json")
        assert GroupConjugacyClassesResult.model_validate(payload) == result

    def test_non_permutation_class_element_rejected(self) -> None:
        """A relayed payload claiming a class element outside the retained
        source group cannot revalidate as an exact conjugacy partition."""
        from jacobian.math.group._models import GroupConjugacyClassesResult

        payload = self._s3_result().model_dump(mode="json")
        identity_class = next(
            entry for entry in payload["classes"] if entry["elements"] == [[0, 1, 2]]
        )
        identity_class["elements"] = [[9, 9, 9]]
        with pytest.raises(ValidationError, match="conjugacy partition"):
            GroupConjugacyClassesResult.model_validate(payload)

    def test_incomplete_partition_rejected(self) -> None:
        from jacobian.math.group._models import GroupConjugacyClassesResult

        payload = self._s3_result().model_dump(mode="json")
        payload["classes"] = payload["classes"][:1]
        payload["class_count"] = 1
        with pytest.raises(ValidationError, match="conjugacy partition"):
            GroupConjugacyClassesResult.model_validate(payload)


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


class TestNativeEnumerationGeneratorCap:
    def test_generator_count_capped_before_permutation_construction(self) -> None:
        from jacobian.math.group.operations import conjugacy_classes, subgroup_lattice

        oversized = [[*range(64)]] * 65
        with pytest.raises(ValueError, match="at most 64 generators"):
            conjugacy_classes(64, oversized)
        with pytest.raises(ValueError, match="at most 64 generators"):
            subgroup_lattice(64, oversized)


class TestNativeConjugacyOrderCap:
    """The exported native kernel mirrors the wire conjugacy-class cap."""

    def test_native_conjugacy_admits_the_prior_order_domain(self) -> None:
        from jacobian.math.group.operations import conjugacy_classes

        classes = conjugacy_classes(5, [[1, 0, 2, 3, 4], [1, 2, 3, 4, 0]])
        assert sum(size for _, size in classes) == 120

    def test_native_conjugacy_rejects_order_above_5000(self) -> None:
        from jacobian.math.group.operations import conjugacy_classes

        with pytest.raises(ValueError, match="exceeds the bounded maximum 5000"):
            conjugacy_classes(7, [[1, 0, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 0]])
