"""Tests for finite geometry operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.finite_geometry._models import (
    GrassmannianCountRequest,
    LinearSubspace,
    PrimeFieldVectorSpace,
    ProjectivePointCanonicalizeRequest,
    ProjectivePointEqualRequest,
    ProjectiveSpaceEnumerateRequest,
    SubspaceComputeRequest,
    SubspaceIntersectionRequest,
    SubspaceMembershipRequest,
    SubspaceSpanRequest,
)
from jacobian.math.finite_geometry._operations import (
    compute_grassmannian_count,
    compute_projective_point_canonicalize,
    compute_projective_point_equal,
    compute_projective_space_enumerate,
    compute_subspace_compute,
    compute_subspace_intersection,
    compute_subspace_membership,
    compute_subspace_span,
)
from jacobian.math.finite_geometry._tools import TOOLS


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "finite_geometry.grassmannian.count",
        "finite_geometry.projective_point.canonicalize",
        "finite_geometry.projective_point.equal.decide",
        "finite_geometry.projective_space.enumerate_points",
        "finite_geometry.subspace.compute",
        "finite_geometry.subspace.intersection.compute",
        "finite_geometry.subspace.membership.decide",
        "finite_geometry.subspace.span.compute",
    }


def test_projective_point_canonicalize_scales_to_one() -> None:
    request = ProjectivePointCanonicalizeRequest(
        space={"field_order": 5, "axis": ("x", "y")}, vector=(2, 3)
    )
    result = compute_projective_point_canonicalize(request)
    assert result.point.coordinates[0] == 1
    assert result.point.space == request.space
    assert result.scale == 2


def test_projective_point_canonicalize_rejects_zero() -> None:
    with pytest.raises(ValidationError) as error:
        ProjectivePointCanonicalizeRequest(
            space={"field_order": 5, "axis": ("x", "y")}, vector=(0, 0)
        )
    assert error.value.errors()[0]["type"] == "finite_geometry.projective_vector_zero"


def test_projective_point_equal_same_point() -> None:
    point = compute_projective_point_canonicalize(
        ProjectivePointCanonicalizeRequest(
            space={"field_order": 5, "axis": ("x", "y")}, vector=(2, 3)
        )
    ).point
    request = ProjectivePointEqualRequest(point_a=point, point_b=point)
    result = compute_projective_point_equal(request)
    assert result.equal is True


def test_projective_point_equal_different_points() -> None:
    space = {"field_order": 5, "axis": ("x", "y")}
    request = ProjectivePointEqualRequest(
        point_a={"space": space, "coordinates": (1, 0)},
        point_b={"space": space, "coordinates": (0, 1)},
    )
    result = compute_projective_point_equal(request)
    assert result.equal is False


def test_subspace_compute_basic() -> None:
    request = SubspaceComputeRequest(
        space={"field_order": 3, "axis": ("x", "y", "z")},
        vectors=((1, 0, 0), (0, 1, 0)),
    )
    result = compute_subspace_compute(request)
    assert result.subspace.dimension == 2
    assert result.subspace.space == request.space


def test_subspace_membership_member() -> None:
    subspace = compute_subspace_compute(
        SubspaceComputeRequest(
            space={"field_order": 3, "axis": ("x", "y", "z")},
            vectors=((1, 0, 0), (0, 1, 0)),
        )
    ).subspace
    request = SubspaceMembershipRequest(subspace=subspace, vector=(1, 1, 0))
    result = compute_subspace_membership(request)
    assert result.is_member is True


def test_subspace_membership_nonmember() -> None:
    subspace = LinearSubspace(
        space={"field_order": 3, "axis": ("x", "y", "z")},
        basis=((1, 0, 0), (0, 1, 0)),
    )
    request = SubspaceMembershipRequest(subspace=subspace, vector=(1, 1, 1))
    result = compute_subspace_membership(request)
    assert result.is_member is False


def test_subspace_span_dependent() -> None:
    request = SubspaceSpanRequest(
        space={"field_order": 2, "axis": ("x", "y")},
        vectors=((1, 0), (1, 0)),
        subspaces=(),
    )
    result = compute_subspace_span(request)
    assert result.subspace.dimension == 1


def test_subspace_intersection_trivial() -> None:
    space = {"field_order": 2, "axis": ("x", "y")}
    request = SubspaceIntersectionRequest(
        subspace_a={"space": space, "basis": ((1, 0),)},
        subspace_b={"space": space, "basis": ((0, 1),)},
    )
    result = compute_subspace_intersection(request)
    assert result.subspace.dimension == 0


def test_subspace_intersection_identical() -> None:
    """Two identical subspaces should intersect at full dimension."""
    space = {"field_order": 2, "axis": ("x", "y")}
    request = SubspaceIntersectionRequest(
        subspace_a={"space": space, "basis": ((1, 0),)},
        subspace_b={"space": space, "basis": ((1, 0),)},
    )
    result = compute_subspace_intersection(request)
    assert result.subspace.dimension == 1


def test_subspace_intersection_overlapping() -> None:
    """Two planes in F_3^3 meeting in a line."""
    space = {"field_order": 3, "axis": ("x", "y", "z")}
    request = SubspaceIntersectionRequest(
        subspace_a={"space": space, "basis": ((1, 0, 0), (0, 1, 0))},
        subspace_b={"space": space, "basis": ((0, 1, 0), (0, 0, 1))},
    )
    result = compute_subspace_intersection(request)
    assert result.subspace.dimension == 1


def test_projective_point_equal_reports_scale() -> None:
    """Scale should be the actual scalar relating the two vectors."""
    point = compute_projective_point_canonicalize(
        ProjectivePointCanonicalizeRequest(
            space={"field_order": 5, "axis": ("x", "y")}, vector=(2, 3)
        )
    ).point
    request = ProjectivePointEqualRequest(point_a=point, point_b=point)
    result = compute_projective_point_equal(request)
    assert result.equal is True
    assert result.point_a == result.point_b


def test_grassmannian_count_lines_in_pg_2_2() -> None:
    request = GrassmannianCountRequest(
        field_order=2, ambient_dimension=3, subspace_dimension=1
    )
    result = compute_grassmannian_count(request)
    assert result.count == 7


def test_grassmannian_count_planes_in_f2_4() -> None:
    request = GrassmannianCountRequest(
        field_order=2, ambient_dimension=4, subspace_dimension=2
    )
    result = compute_grassmannian_count(request)
    assert result.count == 35


def test_projective_space_enumerate_pg1_f2() -> None:
    request = ProjectiveSpaceEnumerateRequest(
        space={"field_order": 2, "axis": ("x", "y")}
    )
    result = compute_projective_space_enumerate(request)
    assert result.count == 3
    assert len(result.points) == 3


def test_request_rejects_nonprime_field() -> None:
    with pytest.raises(ValidationError) as error:
        ProjectivePointCanonicalizeRequest(
            space={"field_order": 4, "axis": ("x", "y")}, vector=(1, 2)
        )
    assert error.value.errors()[0]["type"] == "finite_geometry.field_order_not_prime"


def test_canonical_values_compose_and_reject_different_parents() -> None:
    space = PrimeFieldVectorSpace(field_order=3, axis=("x", "y"))
    computed = compute_subspace_compute(
        SubspaceComputeRequest(space=space, vectors=((1, 0),))
    ).subspace
    assert compute_subspace_membership(
        SubspaceMembershipRequest(subspace=computed, vector=(2, 0))
    ).is_member
    assert (
        compute_subspace_span(
            SubspaceSpanRequest(space=space, vectors=(), subspaces=(computed,))
        ).subspace
        == computed
    )

    other = LinearSubspace(
        space={"field_order": 5, "axis": ("x", "y")}, basis=((1, 0),)
    )
    with pytest.raises(ValidationError) as error:
        SubspaceIntersectionRequest(subspace_a=computed, subspace_b=other)
    assert (
        error.value.errors()[0]["type"]
        == "finite_geometry.intersection_parent_mismatch"
    )


def test_axis_identity_is_part_of_the_parent() -> None:
    point_x = {"space": {"field_order": 3, "axis": ("x", "y")}, "coordinates": (1, 0)}
    point_y = {"space": {"field_order": 3, "axis": ("y", "x")}, "coordinates": (1, 0)}
    with pytest.raises(ValidationError) as error:
        ProjectivePointEqualRequest(point_a=point_x, point_b=point_y)
    assert (
        error.value.errors()[0]["type"] == "finite_geometry.projective_parent_mismatch"
    )


def test_source_bound_results_reject_forged_values() -> None:
    result = compute_subspace_compute(
        SubspaceComputeRequest(
            space={"field_order": 3, "axis": ("x", "y")},
            vectors=((1, 0),),
        )
    )
    payload = result.model_dump()
    payload["subspace"]["basis"] = ((0, 1),)
    with pytest.raises(ValidationError) as error:
        type(result).model_validate(payload)
    assert error.value.errors()[0]["type"] == "finite_geometry.subspace_replay_mismatch"

    count = compute_grassmannian_count(
        GrassmannianCountRequest(
            field_order=2, ambient_dimension=3, subspace_dimension=1
        )
    )
    payload = count.model_dump()
    payload["count"] = 8
    with pytest.raises(ValidationError) as error:
        type(count).model_validate(payload)
    assert (
        error.value.errors()[0]["type"] == "finite_geometry.grassmannian_count_mismatch"
    )
