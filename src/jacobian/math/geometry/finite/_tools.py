"""Finite geometry operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.finite._models import (
    CosetIntersectionProfileRequest,
    CosetIntersectionProfileResult,
    GrassmannianCountRequest,
    GrassmannianCountResult,
    PrimeFieldAffinePlaneRequest,
    PrimeFieldAffinePlaneResult,
    ProjectivePointCanonicalizeRequest,
    ProjectivePointCanonicalizeResult,
    ProjectivePointEqualRequest,
    ProjectivePointEqualResult,
    ProjectiveSpaceEnumerateRequest,
    ProjectiveSpaceEnumerateResult,
    SubspaceComputeRequest,
    SubspaceComputeResult,
    SubspaceIntersectionRequest,
    SubspaceIntersectionResult,
    SubspaceMembershipRequest,
    SubspaceMembershipResult,
    SubspaceSpanRequest,
    SubspaceSpanResult,
)
from jacobian.math.geometry.finite.operations import (
    coset_intersection_profile,
    grassmannian_count,
    prime_field_affine_plane,
    projective_point_canonicalize,
    projective_point_equal,
    projective_space_enumerate,
    subspace_compute,
    subspace_intersection,
    subspace_membership,
    subspace_span,
)


def _compute_projective_point_canonicalize(
    request: ProjectivePointCanonicalizeRequest,
) -> ProjectivePointCanonicalizeResult:
    return projective_point_canonicalize(request.space, request.vector)


def _compute_projective_point_equal(
    request: ProjectivePointEqualRequest,
) -> ProjectivePointEqualResult:
    return projective_point_equal(request.point_a, request.point_b)


def _compute_subspace_compute(
    request: SubspaceComputeRequest,
) -> SubspaceComputeResult:
    return subspace_compute(request.space, request.vectors)


def _compute_subspace_membership(
    request: SubspaceMembershipRequest,
) -> SubspaceMembershipResult:
    return subspace_membership(request.subspace, request.vector)


def _compute_subspace_span(request: SubspaceSpanRequest) -> SubspaceSpanResult:
    return subspace_span(request.space, request.vectors, request.subspaces)


def _compute_subspace_intersection(
    request: SubspaceIntersectionRequest,
) -> SubspaceIntersectionResult:
    return subspace_intersection(request.subspace_a, request.subspace_b)


def _compute_grassmannian_count(
    request: GrassmannianCountRequest,
) -> GrassmannianCountResult:
    return grassmannian_count(
        request.field_order, request.ambient_dimension, request.subspace_dimension
    )


def _compute_projective_space_enumerate(
    request: ProjectiveSpaceEnumerateRequest,
) -> ProjectiveSpaceEnumerateResult:
    return projective_space_enumerate(request.space)


def _compute_prime_field_affine_plane(
    request: PrimeFieldAffinePlaneRequest,
) -> PrimeFieldAffinePlaneResult:
    return prime_field_affine_plane(request.prime_order)


def _compute_coset_intersection_profile(
    request: CosetIntersectionProfileRequest,
) -> CosetIntersectionProfileResult:
    return coset_intersection_profile(request.space, request.subspace, request.subset)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="finite_geometry.projective_point.canonicalize",
        title="Canonicalize a projective point",
        description="Scale a nonzero finite-field vector so its first nonzero coordinate "
        "is one, returning the canonical projective point representative and "
        "the scale factor.",
        request_type=ProjectivePointCanonicalizeRequest,
        result_type=ProjectivePointCanonicalizeResult,
        run=_compute_projective_point_canonicalize,
        tags=("finite-geometry", "projective-point", "exact"),
        examples=(
            OperationExample(
                name="fp2_point",
                description="Canonicalize [2,3] in F_5^2.",
                input={
                    "space": {"field_order": 5, "axis": ["x", "y"]},
                    "vector": [2, 3],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="finite_geometry.projective_point.equal.decide",
        title="Decide whether two vectors define the same projective point",
        description="Check whether two nonzero finite-field vectors are nonzero scalar "
        "multiples of each other.",
        request_type=ProjectivePointEqualRequest,
        result_type=ProjectivePointEqualResult,
        run=_compute_projective_point_equal,
        tags=("finite-geometry", "projective-point", "exact"),
        examples=(
            OperationExample(
                name="equal_points",
                description="Check [2,3] and [4,1] in F_5^2 are the same projective point.",
                input={
                    "point_a": {
                        "space": {"field_order": 5, "axis": ["x", "y"]},
                        "coordinates": [1, 4],
                    },
                    "point_b": {
                        "space": {"field_order": 5, "axis": ["x", "y"]},
                        "coordinates": [1, 4],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="finite_geometry.subspace.compute",
        title="Compute the canonical basis of a subspace",
        description="Compute the canonical RREF basis, dimension, and ambient dimension "
        "of the linear span of a family of vectors over a prime field.",
        request_type=SubspaceComputeRequest,
        result_type=SubspaceComputeResult,
        run=_compute_subspace_compute,
        tags=("finite-geometry", "subspace", "exact"),
        examples=(
            OperationExample(
                name="plane_in_f3",
                description="Compute the span of [1,0,0] and [0,1,0] in F_3^3.",
                input={
                    "space": {"field_order": 3, "axis": ["x", "y", "z"]},
                    "vectors": [[1, 0, 0], [0, 1, 0]],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="finite_geometry.subspace.membership.decide",
        title="Decide subspace membership",
        description="Check whether a word lies in the row space of the given generators "
        "over a prime field.",
        request_type=SubspaceMembershipRequest,
        result_type=SubspaceMembershipResult,
        run=_compute_subspace_membership,
        tags=("finite-geometry", "subspace", "exact"),
        examples=(
            OperationExample(
                name="member_word",
                description="Check [1,1,0] is in span{[1,0,0],[0,1,0]} in F_3.",
                input={
                    "subspace": {
                        "space": {"field_order": 3, "axis": ["x", "y", "z"]},
                        "basis": [[1, 0, 0], [0, 1, 0]],
                    },
                    "vector": [1, 1, 0],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="finite_geometry.subspace.span.compute",
        title="Compute the span of vectors",
        description="Return the exact linear span of labelled points/subspaces over a "
        "prime field with canonical RREF basis and dimension.",
        request_type=SubspaceSpanRequest,
        result_type=SubspaceSpanResult,
        run=_compute_subspace_span,
        tags=("finite-geometry", "subspace", "exact"),
        examples=(
            OperationExample(
                name="span_two_vectors",
                description="Span of [1,0] and [0,1] in F_2.",
                input={
                    "space": {"field_order": 2, "axis": ["x", "y"]},
                    "vectors": [[1, 0], [0, 1]],
                    "subspaces": [],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="finite_geometry.subspace.intersection.compute",
        title="Compute the intersection of two subspaces",
        description="Return the exact canonical basis and dimension of the intersection "
        "of two subspaces given by generator matrices over a prime field.",
        request_type=SubspaceIntersectionRequest,
        result_type=SubspaceIntersectionResult,
        run=_compute_subspace_intersection,
        tags=("finite-geometry", "subspace", "intersection", "exact"),
        examples=(
            OperationExample(
                name="intersection_of_planes",
                description="Intersection of two lines in F_2^2.",
                input={
                    "subspace_a": {
                        "space": {"field_order": 2, "axis": ["x", "y"]},
                        "basis": [[1, 0]],
                    },
                    "subspace_b": {
                        "space": {"field_order": 2, "axis": ["x", "y"]},
                        "basis": [[0, 1]],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="finite_geometry.grassmannian.count",
        title="Count k-dimensional subspaces (Gaussian binomial)",
        description="Compute the exact Gaussian binomial coefficient [n choose k]_q, the "
        "number of k-dimensional subspaces of F_q^n.",
        request_type=GrassmannianCountRequest,
        result_type=GrassmannianCountResult,
        run=_compute_grassmannian_count,
        tags=("finite-geometry", "grassmannian", "exact"),
        examples=(
            OperationExample(
                name="lines_in_f2_3",
                description="Count lines in PG(2, F_2) = [3 choose 1]_2.",
                input={
                    "field_order": 2,
                    "ambient_dimension": 3,
                    "subspace_dimension": 1,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="finite_geometry.projective_space.enumerate_points",
        title="Enumerate all projective points of PG(d, q)",
        description="Enumerate all canonical representatives of the projective space PG(d, "
        "q) over a prime field as one typed point sequence that owns the "
        "declared parent space and serializes each point as a bare canonical "
        "coordinate tuple relative to it.",
        request_type=ProjectiveSpaceEnumerateRequest,
        result_type=ProjectiveSpaceEnumerateResult,
        run=_compute_projective_space_enumerate,
        tags=("finite-geometry", "projective-space", "exact"),
        examples=(
            OperationExample(
                name="pg_2_3",
                description="Enumerate all points of PG(1, F_2).",
                input={"space": {"field_order": 2, "axis": ["x", "y"]}},
            ),
        ),
    ),
    MathTool(
        operation_id="finite_geometry.affine_plane.prime_field.construct",
        title="Construct the prime-field affine plane AG(2, q)",
        description="Construct the complete affine plane AG(2, q) over a prime field: "
        "q^2 labelled points, q(q+1) labelled lines, exact point-line "
        "incidences, and q+1 parallel classes partitioning the line axis.",
        request_type=PrimeFieldAffinePlaneRequest,
        result_type=PrimeFieldAffinePlaneResult,
        run=_compute_prime_field_affine_plane,
        tags=("finite-geometry", "affine-plane", "exact"),
        examples=(
            OperationExample(
                name="ag_2_2",
                description="Construct the affine plane AG(2, 2).",
                input={"prime_order": 2},
            ),
        ),
    ),
    MathTool(
        operation_id="finite_geometry.subspace.coset_intersection_profile.compute",
        title="Partition a finite subset by subspace cosets",
        description=(
            "For a prime-field vector space V, a linear subspace H in RREF, and "
            "a duplicate-free finite subset A of V, return the complete "
            "partition of A by affine cosets of H. Each occupied coset row "
            "retains its canonical quotient representative with zeros in every "
            "pivot column of H, the sorted members of A in that coset, and "
            "their count. Rows are sorted by representative. The operation "
            f"admits at most {1024:,} subset elements, dimension 32, "
            "field order 10,000, and one bounded quotient-reduction pass."
        ),
        request_type=CosetIntersectionProfileRequest,
        result_type=CosetIntersectionProfileResult,
        run=_compute_coset_intersection_profile,
        tags=(
            "finite-geometry",
            "subspace",
            "coset",
            "partition",
            "affine",
            "quotient",
            "exact",
        ),
        examples=(
            OperationExample(
                name="f2_3_two_cosets",
                description=(
                    "Partition {(0,0,0), (1,0,0), (0,1,0)} in F_2^3 by the line "
                    "H=span{(1,0,0)}; the subset has three vectors and occupies "
                    "two cosets. The space axis must be identifiers and the "
                    "subspace must be in RREF with the same field and axis."
                ),
                input={
                    "space": {"field_order": 2, "axis": ["x", "y", "z"]},
                    "subspace": {
                        "space": {"field_order": 2, "axis": ["x", "y", "z"]},
                        "basis": [[1, 0, 0]],
                    },
                    "subset": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
