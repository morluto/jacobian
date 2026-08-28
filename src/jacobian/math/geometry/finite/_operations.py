"""Domain functions for finite geometry operations."""

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
    GrassmannianCountRequest,
    GrassmannianCountResult,
    LinearSubspace,
    ParallelClass,
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
from jacobian.math.geometry.finite.values import (
    MAX_DIM,
    ProjectivePoint,
    ProjectivePointSequence,
)


def _domain_error(location: tuple[str | int, ...], code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code=f"finite_geometry.{code}",
        message=message,
    )


def _admit_span(request: SubspaceSpanRequest) -> None:
    if not request.vectors and not request.subspaces:
        _domain_error(
            ("vectors",),
            "span_source_required",
            "span requires at least one vector or subspace",
        )
    if (
        len(request.vectors) + sum(len(item.basis) for item in request.subspaces)
        > MAX_DIM
    ):
        _domain_error(
            ("vectors",),
            "span_generator_count_exceeds_bound",
            "span generator count exceeds bound",
        )


def _admit_grassmannian(request: GrassmannianCountRequest) -> None:
    if not isprime(request.field_order):
        _domain_error(
            ("field_order",), "field_order_not_prime", "field_order must be prime"
        )
    if request.subspace_dimension > request.ambient_dimension:
        _domain_error(
            ("subspace_dimension",),
            "subspace_dimension_exceeds_ambient",
            "subspace dimension cannot exceed ambient dimension",
        )


def _admit_projective_enumeration(request: ProjectiveSpaceEnumerateRequest) -> None:
    q = request.space.field_order
    n = len(request.space.axis)
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
            for label in request.space.axis
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


def compute_projective_point_canonicalize(
    request: ProjectivePointCanonicalizeRequest,
) -> ProjectivePointCanonicalizeResult:
    if all(value == 0 for value in request.vector):
        _domain_error(
            ("vector",),
            "zero_projective_vector",
            "zero vector has no projective point",
        )
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
    _domain_error(
        ("vector",),
        "zero_projective_vector",
        "zero vector has no projective point",
    )


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
    _admit_span(request)
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
    _admit_grassmannian(request)
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
    _admit_projective_enumeration(request)

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


def compute_prime_field_affine_plane(
    request: PrimeFieldAffinePlaneRequest,
) -> PrimeFieldAffinePlaneResult:
    """Construct the complete affine plane AG(2, q) over a prime field.

    Points are (x, y) in lexicographic order (index = x * q + y).  Lines are
    enumerated as L_{m,b} = {(x, mx+b mod q) : x in F_q} for each slope m
    (0..q-1) and intercept b (0..q-1), then vertical lines V_b = {(b, y) :
    y in F_q} for each b (0..q-1).  Parallel classes partition the line axis
    into q+1 ordered classes: q slope classes (m=0..q-1) plus the vertical
    class.
    """
    q = request.prime_order
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
