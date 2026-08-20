"""Tests for finite geometry operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.finite_geometry._models import (
    GrassmannianCountRequest,
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
    request = ProjectivePointCanonicalizeRequest(field_order=5, vector=(2, 3))
    result = compute_projective_point_canonicalize(request)
    assert result.canonical_vector[0] == 1
    assert result.scale == 2


def test_projective_point_canonicalize_rejects_zero() -> None:
    with pytest.raises(ValidationError, match="nonzero"):
        ProjectivePointCanonicalizeRequest(field_order=5, vector=(0, 0))


def test_projective_point_equal_same_point() -> None:
    request = ProjectivePointEqualRequest(
        field_order=5, vector_a=(2, 3), vector_b=(4, 1)
    )
    result = compute_projective_point_equal(request)
    assert result.equal is True


def test_projective_point_equal_different_points() -> None:
    request = ProjectivePointEqualRequest(
        field_order=5, vector_a=(1, 0), vector_b=(0, 1)
    )
    result = compute_projective_point_equal(request)
    assert result.equal is False


def test_subspace_compute_basic() -> None:
    request = SubspaceComputeRequest(field_order=3, vectors=((1, 0, 0), (0, 1, 0)))
    result = compute_subspace_compute(request)
    assert result.dimension == 2
    assert result.ambient_dimension == 3


def test_subspace_membership_member() -> None:
    request = SubspaceMembershipRequest(
        field_order=3,
        generators=((1, 0, 0), (0, 1, 0)),
        word=(1, 1, 0),
    )
    result = compute_subspace_membership(request)
    assert result.is_member is True


def test_subspace_membership_nonmember() -> None:
    request = SubspaceMembershipRequest(
        field_order=3,
        generators=((1, 0, 0), (0, 1, 0)),
        word=(1, 1, 1),
    )
    result = compute_subspace_membership(request)
    assert result.is_member is False


def test_subspace_span_dependent() -> None:
    request = SubspaceSpanRequest(field_order=2, vectors=((1, 0), (1, 0)))
    result = compute_subspace_span(request)
    assert result.dimension == 1


def test_subspace_intersection_trivial() -> None:
    request = SubspaceIntersectionRequest(
        field_order=2,
        generators_a=((1, 0),),
        generators_b=((0, 1),),
    )
    result = compute_subspace_intersection(request)
    assert result.dimension == 0


def test_subspace_intersection_identical() -> None:
    """Two identical subspaces should intersect at full dimension."""
    request = SubspaceIntersectionRequest(
        field_order=2,
        generators_a=((1, 0),),
        generators_b=((1, 0),),
    )
    result = compute_subspace_intersection(request)
    assert result.dimension == 1


def test_subspace_intersection_overlapping() -> None:
    """Two planes in F_3^3 meeting in a line."""
    request = SubspaceIntersectionRequest(
        field_order=3,
        generators_a=((1, 0, 0), (0, 1, 0)),
        generators_b=((0, 1, 0), (0, 0, 1)),
    )
    result = compute_subspace_intersection(request)
    assert result.dimension == 1


def test_projective_point_equal_reports_scale() -> None:
    """Scale should be the actual scalar relating the two vectors."""
    request = ProjectivePointEqualRequest(
        field_order=5,
        vector_a=(2, 3),
        vector_b=(4, 1),
    )
    result = compute_projective_point_equal(request)
    assert result.equal is True
    # 4 = 2 * 2 mod 5, so scale should be 2
    assert result.scale == 2


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
    request = ProjectiveSpaceEnumerateRequest(field_order=2, projective_dimension=1)
    result = compute_projective_space_enumerate(request)
    assert result.count == 3
    assert len(result.points) == 3


def test_request_rejects_nonprime_field() -> None:
    with pytest.raises(ValidationError, match="prime"):
        ProjectivePointCanonicalizeRequest(field_order=4, vector=(1, 2))
