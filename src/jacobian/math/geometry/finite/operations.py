"""Public native constructors over finite-geometry canonical values."""

from __future__ import annotations

import unicodedata
from itertools import product
from typing import NoReturn

from sympy import isprime

from jacobian.canonical import encode_strict_json, format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.designs.incidence_structures._models import (
    IncidenceStructure,
)
from jacobian.math.geometry.finite._linear import (
    canonical_basis,
    intersection_basis,
    rref_rank,
)
from jacobian.math.geometry.finite._models import (
    _PROJECTIVE_ENUMERATION_ENVELOPE_BYTES,
    MAX_AFFINE_PLANE_FIELD_ORDER,
    MAX_PROJECTIVE_ENUMERATION_RESULT_BYTES,
    MAX_PROJECTIVE_SPACE_ENUMERATION_VECTORS,
    GrassmannianCountResult,
    LinearSubspace,
    ParallelClass,
    PrimeFieldAffinePlaneResult,
    ProjectivePointCanonicalizeResult,
    ProjectivePointEqualResult,
    ProjectiveSpaceEnumerateResult,
    SubspaceComputeResult,
    SubspaceIntersectionResult,
    SubspaceMembershipResult,
    SubspaceSpanResult,
)
from jacobian.math.geometry.finite.values import (
    MAX_DIM,
    PrimeFieldVectorSpace,
    ProjectivePoint,
    ProjectivePointSequence,
    _validate_vector,
)


def _domain_error(location: tuple[str | int, ...], code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code=f"finite_geometry.{code}",
        message=message,
    )


__all__ = [
    "grassmannian_count",
    "prime_field_affine_plane",
    "projective_point",
    "projective_point_canonicalize",
    "projective_point_equal",
    "projective_space_enumerate",
    "subspace_compute",
    "subspace_intersection",
    "subspace_membership",
    "subspace_span",
]


def projective_point(
    space: PrimeFieldVectorSpace, vector: tuple[int, ...]
) -> ProjectivePoint:
    """Canonicalize one nonzero finite-field vector into its point.

    The vector must hold canonical field residues of ``space``; scaling its
    first nonzero entry to one returns the canonical projective
    representative bound to ``space``.
    """

    _validate_vector(vector, space)
    scale = next((value for value in vector if value != 0), None)
    if scale is None:
        _domain_error(
            ("vector",),
            "projective_vector_zero",
            "zero vector has no projective point",
        )
    inverse = pow(scale, -1, space.field_order)
    return ProjectivePoint(
        space=space,
        coordinates=tuple(value * inverse % space.field_order for value in vector),
    )


def _admit_span(
    vectors: tuple[tuple[int, ...], ...], subspaces: tuple[LinearSubspace, ...]
) -> None:
    if not vectors and not subspaces:
        _domain_error(
            ("vectors",),
            "span_source_required",
            "span requires at least one vector or subspace",
        )
    if len(vectors) + sum(len(item.basis) for item in subspaces) > MAX_DIM:
        _domain_error(
            ("vectors",),
            "span_generator_count_exceeds_bound",
            "span generator count exceeds bound",
        )


def _admit_grassmannian(
    field_order: int, ambient_dimension: int, subspace_dimension: int
) -> None:
    if not isprime(field_order):
        _domain_error(
            ("field_order",), "field_order_not_prime", "field_order must be prime"
        )
    if subspace_dimension > ambient_dimension:
        _domain_error(
            ("subspace_dimension",),
            "subspace_dimension_exceeds_ambient",
            "subspace dimension cannot exceed ambient dimension",
        )


def _admit_projective_enumeration(space: PrimeFieldVectorSpace) -> None:
    q = space.field_order
    n = len(space.axis)
    if q**n > MAX_PROJECTIVE_SPACE_ENUMERATION_VECTORS:
        _domain_error(
            ("space",),
            "enumeration_vector_count_exceeds_bound",
            "projective space exceeds the "
            f"{MAX_PROJECTIVE_SPACE_ENUMERATION_VECTORS}-vector enumeration envelope",
        )
    digit_width = len(str(q - 1))
    point_count = (q**n - 1) // (q - 1)
    per_point_bytes = 2 + n * digit_width + (n - 1) + 1
    predicted = (
        _PROJECTIVE_ENUMERATION_ENVELOPE_BYTES
        + sum(
            len(encode_strict_json(unicodedata.normalize("NFC", label)))
            for label in space.axis
        )
        + point_count * per_point_bytes
    )
    if predicted > MAX_PROJECTIVE_ENUMERATION_RESULT_BYTES:
        _domain_error(
            ("space",),
            "enumeration_result_exceeds_bound",
            "the complete serialized point list would exceed the "
            f"{MAX_PROJECTIVE_ENUMERATION_RESULT_BYTES}-byte result budget",
        )


def projective_point_canonicalize(
    space: PrimeFieldVectorSpace,
    vector: tuple[int, ...],
) -> ProjectivePointCanonicalizeResult:
    point = projective_point(space, vector)
    scale = next(value for value in vector if value != 0)
    return ProjectivePointCanonicalizeResult._from_kernel(
        space=space,
        vector=vector,
        point=point,
        scale=scale,
    )


def projective_point_equal(
    point_a: ProjectivePoint,
    point_b: ProjectivePoint,
) -> ProjectivePointEqualResult:
    if point_a.space != point_b.space:
        _domain_error(
            ("point_a", "space"),
            "projective_parent_mismatch",
            "projective points must have the same field and axis",
        )
    return ProjectivePointEqualResult._from_kernel(
        point_a=point_a,
        point_b=point_b,
        equal=point_a.coordinates == point_b.coordinates,
    )


def subspace_compute(
    space: PrimeFieldVectorSpace,
    vectors: tuple[tuple[int, ...], ...],
) -> SubspaceComputeResult:
    for vector in vectors:
        _validate_vector(vector, space)
    matrix = [list(row) for row in vectors]
    basis = canonical_basis(matrix, space.field_order)
    return SubspaceComputeResult._from_kernel(
        space=space, vectors=vectors, subspace=LinearSubspace(space=space, basis=basis)
    )


def subspace_membership(
    subspace: LinearSubspace,
    vector: tuple[int, ...],
) -> SubspaceMembershipResult:
    _validate_vector(vector, subspace.space)
    matrix = [list(row) for row in subspace.basis]
    word = list(vector)
    q = subspace.space.field_order

    _, rank_g = rref_rank([list(r) for r in matrix], q)
    augmented = [list(row) for row in matrix] + [word]
    _, rank_aug = rref_rank(augmented, q)
    is_member = rank_aug == rank_g

    return SubspaceMembershipResult._from_kernel(
        subspace=subspace, vector=vector, is_member=is_member
    )


def subspace_span(
    space: PrimeFieldVectorSpace,
    vectors: tuple[tuple[int, ...], ...],
    subspaces: tuple[LinearSubspace, ...],
) -> SubspaceSpanResult:
    for vector in vectors:
        _validate_vector(vector, space)
    if any(subspace.space != space for subspace in subspaces):
        _domain_error(
            ("subspaces",),
            "span_parent_mismatch",
            "all subspaces must have the declared field and axis",
        )
    _admit_span(vectors, subspaces)
    matrix = [list(row) for row in vectors]
    matrix.extend(list(row) for subspace in subspaces for row in subspace.basis)
    basis = canonical_basis(matrix, space.field_order)
    return SubspaceSpanResult._from_kernel(
        space=space,
        vectors=vectors,
        subspaces=subspaces,
        subspace=LinearSubspace(space=space, basis=basis),
    )


def subspace_intersection(
    subspace_a: LinearSubspace,
    subspace_b: LinearSubspace,
) -> SubspaceIntersectionResult:
    if subspace_a.space != subspace_b.space:
        _domain_error(
            ("subspace_b", "space"),
            "intersection_parent_mismatch",
            "subspaces must have the same field and axis",
        )
    canonical = intersection_basis(
        subspace_a.basis,
        subspace_b.basis,
        subspace_a.space.field_order,
        len(subspace_a.space.axis),
    )
    return SubspaceIntersectionResult._from_kernel(
        subspace_a=subspace_a,
        subspace_b=subspace_b,
        subspace=LinearSubspace(space=subspace_a.space, basis=canonical),
    )


def grassmannian_count(
    field_order: int,
    ambient_dimension: int,
    subspace_dimension: int,
) -> GrassmannianCountResult:
    _admit_grassmannian(field_order, ambient_dimension, subspace_dimension)
    q = field_order
    n = ambient_dimension
    k = subspace_dimension

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
        field_order=field_order,
        ambient_dimension=ambient_dimension,
        subspace_dimension=subspace_dimension,
        count=format_canonical_integer(count),
    )


def projective_space_enumerate(
    space: PrimeFieldVectorSpace,
) -> ProjectiveSpaceEnumerateResult:
    _admit_projective_enumeration(space)

    q = space.field_order
    n = len(space.axis)

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
        sequence=ProjectivePointSequence(space=space, coordinates=tuple(points)),
    )


def prime_field_affine_plane(
    prime_order: int,
) -> PrimeFieldAffinePlaneResult:
    """Construct the complete affine plane AG(2, q) over a prime field.

    Points are (x, y) in lexicographic order (index = x * q + y).  Lines are
    enumerated as L_{m,b} = {(x, mx+b mod q) : x in F_q} for each slope m
    (0..q-1) and intercept b (0..q-1), then vertical lines V_b = {(b, y) :
    y in F_q} for each b (0..q-1).  Parallel classes partition the line axis
    into q+1 ordered classes: q slope classes (m=0..q-1) plus the vertical
    class.
    """
    q = prime_order
    if not isprime(q):
        _domain_error(
            ("prime_order",), "prime_order_not_prime", "prime_order must be prime"
        )
    if q > MAX_AFFINE_PLANE_FIELD_ORDER:
        _domain_error(
            ("prime_order",),
            "prime_order_exceeds_bound",
            "prime_order exceeds the affine-plane operation envelope",
        )

    # Points: (x, y) in lexicographic order, index = x * q + y
    points = tuple(f"{x},{y}" for x in range(q) for y in range(q))

    # Lines: L_{m,b} for slope m, intercept b, then V_b for vertical b
    block_ids: list[str] = []
    blocks: list[tuple[str, ...]] = []

    # Non-vertical lines: L_{m,b}
    for m in range(q):
        for b in range(q):
            line_id = f"L_{m},{b}"
            block_ids.append(line_id)
            members = tuple(f"{x},{(m * x + b) % q}" for x in range(q))
            blocks.append(members)

    # Vertical lines: V_b
    for b_val in range(q):
        line_id = f"V_{b_val}"
        block_ids.append(line_id)
        members = tuple(f"{b_val},{y}" for y in range(q))
        blocks.append(members)

    incidence = IncidenceStructure(
        points=tuple(points),
        block_ids=tuple(block_ids),
        blocks=tuple(blocks),
    )

    # Parallel classes: q slope classes (m=0..q-1), then 1 vertical class
    parallel_classes: list[ParallelClass] = []
    line_index = 0
    for m in range(q):
        line_ids = tuple(range(line_index, line_index + q))
        parallel_classes.append(
            ParallelClass(
                line_ids=line_ids,
                label=f"slope_{m}",
            )
        )
        line_index += q
    # Vertical class
    vertical_line_ids = tuple(range(line_index, line_index + q))
    parallel_classes.append(
        ParallelClass(
            line_ids=vertical_line_ids,
            label="vertical",
        )
    )

    total_incidences = q * q * (q + 1)

    return PrimeFieldAffinePlaneResult.model_construct(
        prime_order=q,
        incidence=incidence,
        parallel_classes=tuple(parallel_classes),
        total_incidences=total_incidences,
    )
