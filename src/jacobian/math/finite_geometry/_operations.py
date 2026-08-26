"""Domain functions for finite geometry operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.finite_geometry._linear import (
    canonical_basis,
    intersection_basis,
    rref_rank,
)
from jacobian.math.finite_geometry._models import (
    GrassmannianCountRequest,
    GrassmannianCountResult,
    LinearSubspace,
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
from jacobian.math.finite_geometry.values import (
    ProjectivePoint,
    ProjectivePointSequence,
)


def compute_projective_point_canonicalize(
    request: ProjectivePointCanonicalizeRequest,
) -> ProjectivePointCanonicalizeResult:
    vector = list(request.vector)
    q = request.space.field_order
    for _i, v in enumerate(vector):
        if v % q != 0:
            scale = v % q
            inv = pow(scale, -1, q)
            canonical = [(v * inv) % q for v in vector]
            return ProjectivePointCanonicalizeResult._from_kernel(
                request,
                ProjectivePoint(space=request.space, coordinates=tuple(canonical)),
                scale,
            )
    raise ValueError("zero vector has no projective point")


def _canonicalize_projective(vector: list[int], q: int) -> tuple[int, ...]:
    """Canonicalize a nonzero vector by scaling first nonzero entry to 1."""
    for i in range(len(vector)):
        if vector[i] % q != 0:
            inv = pow(vector[i] % q, -1, q)
            return tuple((v * inv) % q for v in vector)
    raise ValueError("zero vector has no projective point")


def compute_projective_point_equal(
    request: ProjectivePointEqualRequest,
) -> ProjectivePointEqualResult:
    return ProjectivePointEqualResult._from_kernel(
        request, request.point_a.coordinates == request.point_b.coordinates
    )


def compute_subspace_compute(
    request: SubspaceComputeRequest,
) -> SubspaceComputeResult:
    matrix = [list(row) for row in request.vectors]
    basis = canonical_basis(matrix, request.space.field_order)
    return SubspaceComputeResult._from_kernel(
        request, LinearSubspace(space=request.space, basis=basis)
    )


def compute_subspace_membership(
    request: SubspaceMembershipRequest,
) -> SubspaceMembershipResult:
    matrix = [list(row) for row in request.subspace.basis]
    word = list(request.vector)
    q = request.subspace.space.field_order

    _, rank_g = rref_rank([list(r) for r in matrix], q)
    augmented = [list(row) for row in matrix] + [word]
    _, rank_aug = rref_rank(augmented, q)
    is_member = rank_aug == rank_g

    return SubspaceMembershipResult._from_kernel(request, is_member)


def compute_subspace_span(
    request: SubspaceSpanRequest,
) -> SubspaceSpanResult:
    matrix = [list(row) for row in request.vectors]
    matrix.extend(list(row) for subspace in request.subspaces for row in subspace.basis)
    basis = canonical_basis(matrix, request.space.field_order)
    return SubspaceSpanResult._from_kernel(
        request, LinearSubspace(space=request.space, basis=basis)
    )


def compute_subspace_intersection(
    request: SubspaceIntersectionRequest,
) -> SubspaceIntersectionResult:
    canonical = intersection_basis(
        request.subspace_a.basis,
        request.subspace_b.basis,
        request.subspace_a.space.field_order,
        len(request.subspace_a.space.axis),
    )
    return SubspaceIntersectionResult._from_kernel(
        request, LinearSubspace(space=request.subspace_a.space, basis=canonical)
    )


def compute_grassmannian_count(
    request: GrassmannianCountRequest,
) -> GrassmannianCountResult:
    q = request.field_order
    n = request.ambient_dimension
    k = request.subspace_dimension

    # Gaussian binomial coefficient: [n choose k]_q
    # = product_{i=0}^{k-1} (q^(n-i) - 1) / (q^(k-i) - 1)
    # But we need exact integer division.
    numerator = 1
    denominator = 1
    for i in range(k):
        numerator *= q ** (n - i) - 1
        denominator *= q ** (k - i) - 1
    count = numerator // denominator
    return GrassmannianCountResult._from_kernel(
        request, format_canonical_integer(count)
    )


def compute_projective_space_enumerate(
    request: ProjectiveSpaceEnumerateRequest,
) -> ProjectiveSpaceEnumerateResult:
    from itertools import product

    q = request.space.field_order
    n = len(request.space.axis)

    seen: dict[tuple[int, ...], bool] = {}
    points: list[tuple[int, ...]] = []

    for vec in product(range(q), repeat=n):
        if all(v == 0 for v in vec):
            continue
        # Canonicalize: scale so first nonzero coordinate is 1
        for i in range(n):
            if vec[i] != 0:
                inv = pow(vec[i], -1, q)
                canonical = tuple((v * inv) % q for v in vec)
                if canonical not in seen:
                    seen[canonical] = True
                    points.append(canonical)
                break

    return ProjectiveSpaceEnumerateResult(
        sequence=ProjectivePointSequence(
            space=request.space, coordinates=tuple(points)
        ),
    )


def verify_projective_point_canonicalize_result(
    result: ProjectivePointCanonicalizeResult,
) -> bool:
    """Verify one bounded externally supplied projective-point claim."""

    scale = next(value for value in result.vector if value != 0)
    expected = tuple(
        value * pow(scale, -1, result.space.field_order) % result.space.field_order
        for value in result.vector
    )
    return result.scale == scale and result.point.coordinates == expected


def verify_projective_point_equal_result(result: ProjectivePointEqualResult) -> bool:
    """Verify one bounded externally supplied point-equality claim."""

    return result.equal == (result.point_a.coordinates == result.point_b.coordinates)


def verify_subspace_compute_result(result: SubspaceComputeResult) -> bool:
    """Verify one bounded externally supplied span-basis claim."""

    return result.subspace.basis == canonical_basis(
        [list(vector) for vector in result.vectors], result.space.field_order
    )


def verify_subspace_membership_result(result: SubspaceMembershipResult) -> bool:
    """Verify one bounded externally supplied subspace-membership claim."""

    q = result.subspace.space.field_order
    _, rank_subspace = rref_rank([list(row) for row in result.subspace.basis], q)
    _, rank_enlarged = rref_rank(
        [*map(list, result.subspace.basis), list(result.vector)], q
    )
    return result.is_member == (rank_subspace == rank_enlarged)


def verify_subspace_span_result(result: SubspaceSpanResult) -> bool:
    """Verify one bounded externally supplied mixed-generator span claim."""

    generators = [*map(list, result.vectors)]
    generators.extend(
        list(row) for subspace in result.subspaces for row in subspace.basis
    )
    return result.subspace.basis == canonical_basis(
        generators, result.space.field_order
    )


def verify_subspace_intersection_result(result: SubspaceIntersectionResult) -> bool:
    """Verify one bounded externally supplied subspace-intersection claim."""

    space = result.subspace_a.space
    return result.subspace.basis == intersection_basis(
        result.subspace_a.basis,
        result.subspace_b.basis,
        space.field_order,
        len(space.axis),
    )


def verify_grassmannian_count_result(result: GrassmannianCountResult) -> bool:
    """Verify one bounded externally supplied Gaussian-binomial claim."""

    numerator = denominator = 1
    for index in range(result.subspace_dimension):
        numerator *= result.field_order ** (result.ambient_dimension - index) - 1
        denominator *= result.field_order ** (result.subspace_dimension - index) - 1
    return result.count == format_canonical_integer(numerator // denominator)
