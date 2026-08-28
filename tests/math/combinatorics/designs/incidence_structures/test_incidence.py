"""Tests for incidence structure operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.designs.incidence_structures._models import (
    ComplementRequest,
    ContainmentProfileRequest,
    DerivedResidualRequest,
    DualRequest,
    GramRequest,
    IncidenceMatrixRequest,
    IncidenceStructure,
    IntersectionsRequest,
    LeviGraphRequest,
    RestrictionRequest,
)
from jacobian.math.combinatorics.designs.incidence_structures._operations import (
    compute_complement,
    compute_containment_profile,
    compute_degree_profile,
    compute_derived_residual,
    compute_dual,
    compute_gram,
    compute_incidence_matrix,
    compute_intersections,
    compute_levi_graph,
    compute_restriction,
)

STRUCTURE = IncidenceStructure.model_validate(
    {
        "points": ["p1", "p2", "p3"],
        "block_ids": ["b1", "b2"],
        "blocks": [["p1", "p2"], ["p2", "p3"]],
    }
)

FANO = IncidenceStructure.model_validate(
    {
        "points": ["1", "2", "3", "4", "5", "6", "7"],
        "block_ids": [
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
            "L6",
            "L7",
        ],
        "blocks": [
            ["1", "2", "3"],
            ["1", "4", "5"],
            ["1", "6", "7"],
            ["2", "4", "6"],
            ["2", "5", "7"],
            ["3", "4", "7"],
            ["3", "5", "6"],
        ],
    }
)


class TestIncidenceMatrix:
    def test_matrix(self) -> None:
        result = compute_incidence_matrix(IncidenceMatrixRequest(incidence=STRUCTURE))
        assert result.matrix == ((1, 0), (1, 1), (0, 1))
        assert result.points == ("p1", "p2", "p3")
        assert result.block_ids == ("b1", "b2")

    def test_duplicate_points_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IncidenceMatrixRequest.model_validate(
                {
                    "incidence": {
                        "points": ["p1", "p1"],
                        "block_ids": ["b1"],
                        "blocks": [["p1"]],
                    }
                }
            )

    def test_validation_uses_stable_error_code(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            IncidenceMatrixRequest.model_validate(
                {
                    "incidence": {
                        "points": ["p1", "p1"],
                        "block_ids": ["b1"],
                        "blocks": [["p1"]],
                    }
                }
            )
        assert exc_info.value.errors()[0]["type"] == (
            "incidence_structure.point_labels_not_distinct"
        )

    def test_invalid_block_member(self) -> None:
        with pytest.raises(ValidationError):
            IncidenceMatrixRequest.model_validate(
                {
                    "incidence": {
                        "points": ["p1"],
                        "block_ids": ["b1"],
                        "blocks": [["p2"]],
                    }
                }
            )


class TestIncidenceStructureCanonicalization:
    def test_block_members_are_canonicalized_to_point_axis_order(self) -> None:
        structure = IncidenceStructure(
            points=("p1", "p2", "p3"),
            block_ids=("b1",),
            blocks=(("p3", "p1"),),
        )
        assert structure.blocks == (("p1", "p3"),)

    def test_equal_membership_in_different_orders_compares_and_serializes_equal(
        self,
    ) -> None:
        first = IncidenceStructure(
            points=("p1", "p2", "p3"),
            block_ids=("b1", "b2"),
            blocks=(("p2", "p1"), ("p3",)),
        )
        second = IncidenceStructure(
            points=("p1", "p2", "p3"),
            block_ids=("b1", "b2"),
            blocks=(("p1", "p2"), ("p3",)),
        )
        assert first == second
        assert first.model_dump(mode="json") == second.model_dump(mode="json")

    def test_canonicalized_source_still_rejects_undecorated_members(self) -> None:
        with pytest.raises(ValidationError):
            IncidenceStructure(
                points=("p1", "p2"),
                block_ids=("b1",),
                blocks=(("p2", "pX"),),
            )

    def test_empty_block_canonicalizes_to_empty(self) -> None:
        structure = IncidenceStructure(
            points=("p1", "p2"),
            block_ids=("b1",),
            blocks=((),),
        )
        assert structure.blocks == ((),)


class TestDegreeProfile:
    def test_degrees(self) -> None:
        result = compute_degree_profile(IncidenceMatrixRequest(incidence=STRUCTURE))
        assert result.point_degrees == (("p1", 1), ("p2", 2), ("p3", 1))
        assert result.block_degrees == (("b1", 2), ("b2", 2))
        assert result.total_incidences == 4

    def test_single_point(self) -> None:
        result = compute_degree_profile(
            IncidenceMatrixRequest.model_validate(
                {
                    "incidence": {
                        "points": ["p1"],
                        "block_ids": ["b1"],
                        "blocks": [["p1"]],
                    }
                }
            )
        )
        assert result.total_incidences == 1


class TestContainmentProfile:
    def test_t1_triangle(self) -> None:
        result = compute_containment_profile(
            ContainmentProfileRequest(incidence=STRUCTURE, t=1)
        )
        assert result.t == 1
        assert len(result.subset_profile) == 3
        # (p1) -> 1, (p2) -> 2, (p3) -> 1
        assert result.subset_profile == (
            (("p1",), 1),
            (("p2",), 2),
            (("p3",), 1),
        )
        assert result.incidence.points == ("p1", "p2", "p3")
        assert result.total_multiplicity == 4
        assert result.min_multiplicity == 1
        assert result.max_multiplicity == 2
        assert not result.is_constant

    def test_t2_triangle(self) -> None:
        result = compute_containment_profile(
            ContainmentProfileRequest(incidence=STRUCTURE, t=2)
        )
        assert len(result.subset_profile) == 3
        # (p1,p2) in b1, (p2,p3) in b2, (p1,p3) in none
        assert result.subset_profile == (
            (("p1", "p2"), 1),
            (("p1", "p3"), 0),
            (("p2", "p3"), 1),
        )
        assert result.min_multiplicity == 0
        assert result.max_multiplicity == 1
        assert result.total_multiplicity == 2

    def test_t1_fano_constant(self) -> None:
        result = compute_containment_profile(
            ContainmentProfileRequest(incidence=FANO, t=1)
        )
        assert result.is_constant
        assert result.constant_lambda == 3

    def test_t2_fano_constant(self) -> None:
        result = compute_containment_profile(
            ContainmentProfileRequest(incidence=FANO, t=2)
        )
        assert result.is_constant
        assert result.constant_lambda == 1


class TestIntersections:
    def test_triangle_intersections(self) -> None:
        result = compute_intersections(IntersectionsRequest(incidence=STRUCTURE))
        assert len(result.pairwise) == 1
        bid, bid2, inter, size = result.pairwise[0]
        assert bid == "b1"
        assert bid2 == "b2"
        assert size == 1
        assert set(inter) == {"p2"}
        assert result.histogram == ((1, 1),)

    def test_fano_intersections(self) -> None:
        result = compute_intersections(IntersectionsRequest(incidence=FANO))
        # Fano plane: every pair of lines meets in exactly 1 point
        assert len(result.pairwise) == 21  # C(7,2)
        assert all(size == 1 for _, _, _, size in result.pairwise)
        assert result.histogram == ((1, 21),)


class TestDual:
    def test_triangle_dual(self) -> None:
        result = compute_dual(DualRequest(incidence=STRUCTURE))
        # Dual points = block IDs, dual blocks = per original point
        assert result.points == ("b1", "b2")
        assert result.block_ids == ("p1", "p2", "p3")
        # Original p1 in b1 only -> block {b1}
        # Original p2 in b1 and b2 -> block {b1, b2}
        # Original p3 in b2 only -> block {b2}
        assert result.blocks == (("b1",), ("b1", "b2"), ("b2",))

    def test_dual_of_dual(self) -> None:
        first = compute_dual(DualRequest(incidence=STRUCTURE))
        # Build a new structure from the dual and dual again
        second = compute_dual(
            DualRequest.model_validate(
                {
                    "incidence": {
                        "points": list(first.points),
                        "block_ids": list(first.block_ids),
                        "blocks": [list(b) for b in first.blocks],
                    }
                }
            )
        )
        # dual(dual) recovers original up to transport: second dual points
        # are the first dual's block IDs = original points
        assert set(second.points) == {"p1", "p2", "p3"}
        assert set(second.block_ids) == {"b1", "b2"}


class TestComplement:
    def test_triangle_complement(self) -> None:
        result = compute_complement(ComplementRequest(incidence=STRUCTURE))
        assert result.points == ("p1", "p2", "p3")
        assert result.block_ids == ("b1", "b2")
        # b1 = {p1,p2}, complement = {p3}
        # b2 = {p2,p3}, complement = {p1}
        assert result.blocks == (("p3",), ("p1",))
        assert len(result.correspondence) == 2
        bid, old, new = result.correspondence[0]
        assert bid == "b1"
        assert set(old) == {"p1", "p2"}
        assert set(new) == {"p3"}

    def test_complement_size_identity(self) -> None:
        """Complement maps block size k to v-k."""
        result = compute_complement(ComplementRequest(incidence=STRUCTURE))
        v = len(STRUCTURE.points)
        # b1 size 2 -> complement size v-2 = 1
        # b2 size 2 -> complement size v-2 = 1
        assert len(result.blocks[0]) == v - 2
        assert len(result.blocks[1]) == v - 2


class TestRestriction:
    def test_restrict_points(self) -> None:
        result = compute_restriction(
            RestrictionRequest(
                incidence=STRUCTURE,
                points=("p1", "p2"),
            )
        )
        assert result.points == ("p1", "p2")
        # b1 = {p1,p2} -> {p1,p2}
        # b2 = {p2,p3} -> {p2}
        assert result.blocks[0] == ("p1", "p2")
        assert result.blocks[1] == ("p2",)

    def test_restrict_blocks(self) -> None:
        result = compute_restriction(
            RestrictionRequest(
                incidence=STRUCTURE,
                block_ids=("b1",),
            )
        )
        assert result.block_ids == ("b1",)
        assert result.blocks == (("p1", "p2"),)
        assert result.points == ("p1", "p2", "p3")

    def test_restrict_both(self) -> None:
        result = compute_restriction(
            RestrictionRequest(
                incidence=STRUCTURE,
                points=("p2",),
                block_ids=("b2",),
            )
        )
        assert result.points == ("p2",)
        assert result.block_ids == ("b2",)
        # b2 = {p2,p3} restricted to {p2} -> {p2}
        assert result.blocks == (("p2",),)


class TestDerivedResidual:
    def test_derived_at_p2(self) -> None:
        result = compute_derived_residual(
            DerivedResidualRequest(incidence=STRUCTURE, point="p2", kind="derived")
        )
        assert result.kind == "derived"
        assert result.anchor_point == "p2"
        assert result.points == ("p1", "p3")
        # Blocks containing p2: b1 = {p1,p2}, b2 = {p2,p3}
        # Remove p2: b1 -> {p1}, b2 -> {p3}
        assert result.block_ids == ("b1", "b2")
        assert result.blocks == (("p1",), ("p3",))
        assert result.source_blocks == ("b1", "b2")

    def test_residual_at_p1(self) -> None:
        result = compute_derived_residual(
            DerivedResidualRequest(incidence=STRUCTURE, point="p1", kind="residual")
        )
        assert result.kind == "residual"
        assert result.anchor_point == "p1"
        assert result.points == ("p2", "p3")
        # Blocks NOT containing p1: only b2 = {p2,p3}
        assert result.block_ids == ("b2",)
        assert result.blocks == (("p2", "p3"),)
        assert result.source_blocks == ("b2",)

    def test_derived_at_nonexistent_point(self) -> None:
        with pytest.raises(ValueError):
            compute_derived_residual(
                DerivedResidualRequest(incidence=STRUCTURE, point="pX")
            )


class TestLeviGraph:
    def test_triangle_levi(self) -> None:
        result = compute_levi_graph(LeviGraphRequest(incidence=STRUCTURE))
        assert result.left_vertices == ("p:p1", "p:p2", "p:p3")
        assert result.right_vertices == ("b:b1", "b:b2")
        assert ("p:p1", "b:b1") in result.edges
        assert ("p:p2", "b:b1") in result.edges
        assert ("p:p2", "b:b2") in result.edges
        assert ("p:p3", "b:b2") in result.edges
        assert len(result.edges) == 4  # 2+2+0+0... wait 2+2 = 4 incidences

    def test_label_collision(self) -> None:
        """Point and block labels with same raw string stay distinct."""
        result = compute_levi_graph(
            LeviGraphRequest.model_validate(
                {
                    "incidence": {
                        "points": ["a"],
                        "block_ids": ["a"],
                        "blocks": [["a"]],
                    }
                }
            )
        )
        assert result.left_vertices == ("p:a",)
        assert result.right_vertices == ("b:a",)
        assert result.edges == (("p:a", "b:a"),)


class TestGram:
    def test_gram_point_axis(self) -> None:
        result = compute_gram(GramRequest(incidence=STRUCTURE, axis="point"))
        assert result.axis == "point"
        assert result.labels == ("p1", "p2", "p3")
        # N = [[1,0],[1,1],[0,1]]
        # N N^T:
        # p1: (1*1+0*0, 1*1+0*1, 0) = (1, 1, 0)
        # p2: (1*1+1*1, 1*1+1*1, 0+1) = wait let me recompute
        # N row p1 = [1, 0], p2 = [1, 1], p3 = [0, 1]
        # N N^T[i][j] = dot(row_i, row_j)
        # (p1,p1)=1, (p1,p2)=1, (p1,p3)=0
        # (p2,p1)=1, (p2,p2)=2, (p2,p3)=1
        # (p3,p1)=0, (p3,p2)=1, (p3,p3)=1
        assert result.matrix == (
            (1, 1, 0),
            (1, 2, 1),
            (0, 1, 1),
        )

    def test_gram_block_axis(self) -> None:
        result = compute_gram(GramRequest(incidence=STRUCTURE, axis="block"))
        assert result.axis == "block"
        assert result.labels == ("b1", "b2")
        # N^T N:
        # col b1 = [1,1,0], col b2 = [0,1,1]
        # (b1,b1)=2, (b1,b2)=1
        # (b2,b1)=1, (b2,b2)=2
        assert result.matrix == (
            (2, 1),
            (1, 2),
        )

    def test_gram_invalid_axis(self) -> None:
        with pytest.raises(ValidationError):
            GramRequest(incidence=STRUCTURE, axis="invalid")
