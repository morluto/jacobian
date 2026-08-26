"""Finite geometry operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.finite_geometry._models import (
    GrassmannianCountRequest,
    GrassmannianCountResult,
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


def _op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "finite_geometry.projective_point.canonicalize",
        "Canonicalize a projective point",
        "Scale a nonzero finite-field vector so its first nonzero coordinate "
        "is one, returning the canonical projective point representative and "
        "the scale factor.",
        ProjectivePointCanonicalizeRequest,
        ProjectivePointCanonicalizeResult,
        compute_projective_point_canonicalize,
        "finite-geometry",
        "projective-point",
        "exact",
        examples=(
            example(
                "fp2_point",
                "Canonicalize [2,3] in F_5^2.",
                {"space": {"field_order": 5, "axis": ["x", "y"]}, "vector": [2, 3]},
            ),
        ),
    ),
    _op(
        "finite_geometry.projective_point.equal.decide",
        "Decide whether two vectors define the same projective point",
        "Check whether two nonzero finite-field vectors are nonzero scalar "
        "multiples of each other.",
        ProjectivePointEqualRequest,
        ProjectivePointEqualResult,
        compute_projective_point_equal,
        "finite-geometry",
        "projective-point",
        "exact",
        examples=(
            example(
                "equal_points",
                "Check [2,3] and [4,1] in F_5^2 are the same projective point.",
                {
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
    _op(
        "finite_geometry.subspace.compute",
        "Compute the canonical basis of a subspace",
        "Compute the canonical RREF basis, dimension, and ambient dimension "
        "of the linear span of a family of vectors over a prime field.",
        SubspaceComputeRequest,
        SubspaceComputeResult,
        compute_subspace_compute,
        "finite-geometry",
        "subspace",
        "exact",
        examples=(
            example(
                "plane_in_f3",
                "Compute the span of [1,0,0] and [0,1,0] in F_3^3.",
                {
                    "space": {"field_order": 3, "axis": ["x", "y", "z"]},
                    "vectors": [[1, 0, 0], [0, 1, 0]],
                },
            ),
        ),
    ),
    _op(
        "finite_geometry.subspace.membership.decide",
        "Decide subspace membership",
        "Check whether a word lies in the row space of the given generators "
        "over a prime field.",
        SubspaceMembershipRequest,
        SubspaceMembershipResult,
        compute_subspace_membership,
        "finite-geometry",
        "subspace",
        "exact",
        examples=(
            example(
                "member_word",
                "Check [1,1,0] is in span{[1,0,0],[0,1,0]} in F_3.",
                {
                    "subspace": {
                        "space": {"field_order": 3, "axis": ["x", "y", "z"]},
                        "basis": [[1, 0, 0], [0, 1, 0]],
                    },
                    "vector": [1, 1, 0],
                },
            ),
        ),
    ),
    _op(
        "finite_geometry.subspace.span.compute",
        "Compute the span of vectors",
        "Return the exact linear span of labelled points/subspaces over a "
        "prime field with canonical RREF basis and dimension.",
        SubspaceSpanRequest,
        SubspaceSpanResult,
        compute_subspace_span,
        "finite-geometry",
        "subspace",
        "exact",
        examples=(
            example(
                "span_two_vectors",
                "Span of [1,0] and [0,1] in F_2.",
                {
                    "space": {"field_order": 2, "axis": ["x", "y"]},
                    "vectors": [[1, 0], [0, 1]],
                    "subspaces": [],
                },
            ),
        ),
    ),
    _op(
        "finite_geometry.subspace.intersection.compute",
        "Compute the intersection of two subspaces",
        "Return the exact canonical basis and dimension of the intersection "
        "of two subspaces given by generator matrices over a prime field.",
        SubspaceIntersectionRequest,
        SubspaceIntersectionResult,
        compute_subspace_intersection,
        "finite-geometry",
        "subspace",
        "intersection",
        "exact",
        examples=(
            example(
                "intersection_of_planes",
                "Intersection of two lines in F_2^2.",
                {
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
    _op(
        "finite_geometry.grassmannian.count",
        "Count k-dimensional subspaces (Gaussian binomial)",
        "Compute the exact Gaussian binomial coefficient [n choose k]_q, the "
        "number of k-dimensional subspaces of F_q^n.",
        GrassmannianCountRequest,
        GrassmannianCountResult,
        compute_grassmannian_count,
        "finite-geometry",
        "grassmannian",
        "exact",
        examples=(
            example(
                "lines_in_f2_3",
                "Count lines in PG(2, F_2) = [3 choose 1]_2.",
                {"field_order": 2, "ambient_dimension": 3, "subspace_dimension": 1},
            ),
        ),
    ),
    _op(
        "finite_geometry.projective_space.enumerate_points",
        "Enumerate all projective points of PG(d, q)",
        "Enumerate all canonical representatives of the projective space PG(d, "
        "q) over a prime field as one typed point sequence that owns the "
        "declared parent space and serializes each point as a bare canonical "
        "coordinate tuple relative to it.",
        ProjectiveSpaceEnumerateRequest,
        ProjectiveSpaceEnumerateResult,
        compute_projective_space_enumerate,
        "finite-geometry",
        "projective-space",
        "exact",
        examples=(
            example(
                "pg_2_3",
                "Enumerate all points of PG(1, F_2).",
                {"space": {"field_order": 2, "axis": ["x", "y"]}},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
