"""Domain functions for finite geometry operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
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
from jacobian.math.prime_field_linear_algebra import (
    PrimeFieldMatrix,
    nullspace,
    rref,
)


def _rref(matrix: list[list[int]], field_order: int) -> tuple[list[list[int]], int]:
    """Reduced row echelon form and rank over a prime field."""
    shared_matrix = PrimeFieldMatrix(
        prime=field_order,
        entries=tuple(tuple(row) for row in matrix),
        columns=len(matrix[0]) if matrix else 0,
    )
    reduced, pivots = rref(shared_matrix)
    return [list(row) for row in reduced], len(pivots)


def _canonical_basis(matrix: list[list[int]], field_order: int) -> list[list[int]]:
    rref, rank = _rref(matrix, field_order)
    return list(rref[:rank]) if rank > 0 else []


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
            return ProjectivePointCanonicalizeResult(
                **request.model_dump(),
                point=ProjectivePoint(
                    space=request.space, coordinates=tuple(canonical)
                ),
                scale=scale,
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
    return ProjectivePointEqualResult(
        point_a=request.point_a,
        point_b=request.point_b,
        equal=request.point_a.coordinates == request.point_b.coordinates,
    )


def compute_subspace_compute(
    request: SubspaceComputeRequest,
) -> SubspaceComputeResult:
    matrix = [list(row) for row in request.vectors]
    basis = _canonical_basis(matrix, request.space.field_order)
    return SubspaceComputeResult(
        **request.model_dump(),
        subspace=LinearSubspace(space=request.space, basis=tuple(map(tuple, basis))),
    )


def compute_subspace_membership(
    request: SubspaceMembershipRequest,
) -> SubspaceMembershipResult:
    matrix = [list(row) for row in request.subspace.basis]
    word = list(request.vector)
    q = request.subspace.space.field_order

    _, rank_g = _rref([list(r) for r in matrix], q)
    augmented = [list(row) for row in matrix] + [word]
    _, rank_aug = _rref(augmented, q)
    is_member = rank_aug == rank_g

    return SubspaceMembershipResult(
        is_member=is_member,
        subspace=request.subspace,
        vector=request.vector,
    )


def compute_subspace_span(
    request: SubspaceSpanRequest,
) -> SubspaceSpanResult:
    matrix = [list(row) for row in request.vectors]
    matrix.extend(list(row) for subspace in request.subspaces for row in subspace.basis)
    basis = _canonical_basis(matrix, request.space.field_order)
    return SubspaceSpanResult(
        **request.model_dump(),
        subspace=LinearSubspace(space=request.space, basis=tuple(map(tuple, basis))),
    )


def compute_subspace_intersection(
    request: SubspaceIntersectionRequest,
) -> SubspaceIntersectionResult:
    canonical = _intersection_basis(request.subspace_a, request.subspace_b)
    return SubspaceIntersectionResult(
        **request.model_dump(),
        subspace=LinearSubspace(space=request.subspace_a.space, basis=canonical),
    )


def _intersection_basis(
    subspace_a: LinearSubspace,
    subspace_b: LinearSubspace,
) -> tuple[tuple[int, ...], ...]:
    q = subspace_a.space.field_order
    matrix_a = [list(row) for row in subspace_a.basis]
    matrix_b = [list(row) for row in subspace_b.basis]

    # Intersect row spaces: solve x in rowspace(A) and x in rowspace(B)
    # Intersection of rowspace(A) ∩ rowspace(B):
    # A vector v is in both row spaces iff v = A^T * a = B^T * b for some
    # coefficient vectors a, b.  This gives [A^T | -B^T] * [a; b] = 0.
    # The nullspace of [A^T | -B^T] gives coefficient pairs [a; b]; the
    # intersection vectors are A^T * a (= B^T * b).
    n = len(subspace_a.space.axis)
    k = len(matrix_a)
    m = len(matrix_b)

    # Build [A^T | -B^T]: an n x (k+m) matrix where columns 0..k-1 are A^T
    # and columns k..k+m-1 are -B^T.
    combined: list[list[int]] = []
    for j in range(n):
        row: list[int] = []
        for i in range(k):
            row.append(matrix_a[i][j] % q)
        for i in range(m):
            row.append((-matrix_b[i][j]) % q)
        combined.append(row)

    null = _nullspace(combined, q)

    intersection_basis: list[list[int]] = []
    for vec in null:
        a_part = vec[:k]
        iv = [0] * n
        for ai, coeff in enumerate(a_part):
            for j in range(n):
                iv[j] = (iv[j] + coeff * matrix_a[ai][j]) % q
        intersection_basis.append(iv)

    # Canonicalize
    canonical = _canonical_basis(intersection_basis, q) if intersection_basis else []

    return tuple(tuple(row) for row in canonical)


def _nullspace(matrix: list[list[int]], field_order: int) -> list[list[int]]:
    shared_matrix = PrimeFieldMatrix(
        prime=field_order,
        entries=tuple(tuple(row) for row in matrix),
        columns=len(matrix[0]),
    )
    return [list(row) for row in nullspace(shared_matrix)]


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
    return GrassmannianCountResult(
        field_order=q,
        ambient_dimension=n,
        subspace_dimension=k,
        count=format_canonical_integer(count),
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
