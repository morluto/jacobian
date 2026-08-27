"""Typed wire contracts for finite geometry operations."""

from __future__ import annotations

import unicodedata
from typing import Self

from pydantic import ConfigDict, Field, model_validator
from sympy import isprime

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json
from jacobian.math.finite_geometry.values import (
    MAX_DIM,
    MAX_FIELD_ORDER,
    PrimeFieldVectorSpace,
    ProjectivePoint,
    ProjectivePointSequence,
    _validate_vector,
    _validation_error,
)
from jacobian.math.incidence_structures._models import IncidenceStructure

MAX_PROJECTIVE_SPACE_ENUMERATION_VECTORS = 65_536
# Owner-local serialized-result budget for the complete enumeration reply,
# kept below Jacobian's 10 MiB canonical transport limit. Admission predicts
# the worst-case canonical bytes from mathematics alone: at most
# (q**n - 1)/(q - 1) points, each encoded as a bare coordinate array whose
# entries carry at most len(str(q - 1)) digits (canonical residues are < q),
# plus the declared parent space echoed once and a fixed key/method header.
# Every admitted request therefore returns its complete declared result
# instead of failing transport validation only after enumeration.
MAX_PROJECTIVE_ENUMERATION_RESULT_BYTES = 8 * 1024 * 1024
# Conservative fixed overhead for result keys, the method string, the count
# digits, enclosing braces, and array punctuation outside per-point entries.
_PROJECTIVE_ENUMERATION_ENVELOPE_BYTES = 256


class LinearSubspace(StrictModel):
    """A subspace represented by its unique RREF basis in an ordered parent."""

    space: PrimeFieldVectorSpace
    basis: tuple[tuple[int, ...], ...] = Field(max_length=32)

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        for row in self.basis:
            _validate_vector(row, self.space)
        pivots: list[int] = []
        for row_index, row in enumerate(self.basis):
            try:
                pivot = next(index for index, value in enumerate(row) if value != 0)
            except StopIteration as exc:
                raise _validation_error(
                    "rref_zero_row", "RREF basis cannot contain a zero row"
                ) from exc
            if row[pivot] != 1 or (pivots and pivot <= pivots[-1]):
                raise _validation_error(
                    "basis_not_rref", "basis must be in reduced row echelon form"
                )
            if any(
                other[pivot] != 0
                for other_index, other in enumerate(self.basis)
                if other_index != row_index
            ):
                raise _validation_error(
                    "basis_not_rref", "basis must be in reduced row echelon form"
                )
            pivots.append(pivot)
        return self

    @property
    def dimension(self) -> int:
        return len(self.basis)


class ProjectivePointCanonicalizeRequest(StrictModel):
    space: PrimeFieldVectorSpace
    vector: tuple[int, ...]

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_vector(self.vector, self.space)
        if all(value == 0 for value in self.vector):
            raise _validation_error(
                "projective_vector_zero", "projective point vector must be nonzero"
            )
        return self


class ProjectivePointEqualRequest(StrictModel):
    point_a: ProjectivePoint
    point_b: ProjectivePoint

    @model_validator(mode="after")
    def require_same_parent(self) -> Self:
        if self.point_a.space != self.point_b.space:
            raise _validation_error(
                "projective_parent_mismatch",
                "projective points must have the same field and axis",
            )
        return self


class SubspaceComputeRequest(StrictModel):
    space: PrimeFieldVectorSpace
    vectors: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        for vector in self.vectors:
            _validate_vector(vector, self.space)
        return self


class SubspaceMembershipRequest(StrictModel):
    subspace: LinearSubspace
    vector: tuple[int, ...]

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_vector(self.vector, self.subspace.space)
        return self


class SubspaceSpanRequest(StrictModel):
    space: PrimeFieldVectorSpace
    vectors: tuple[tuple[int, ...], ...] = Field(max_length=32)
    subspaces: tuple[LinearSubspace, ...] = Field(max_length=32)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if not self.vectors and not self.subspaces:
            raise _validation_error(
                "span_empty_generators", "span requires at least one vector or subspace"
            )
        for vector in self.vectors:
            _validate_vector(vector, self.space)
        if any(subspace.space != self.space for subspace in self.subspaces):
            raise _validation_error(
                "span_parent_mismatch",
                "all subspaces must have the declared field and axis",
            )
        if len(self.vectors) + sum(len(item.basis) for item in self.subspaces) > 32:
            raise _validation_error(
                "span_generator_count_exceeded", "span generator count exceeds bound"
            )
        return self


class SubspaceIntersectionRequest(StrictModel):
    subspace_a: LinearSubspace
    subspace_b: LinearSubspace

    @model_validator(mode="after")
    def require_same_parent(self) -> Self:
        if self.subspace_a.space != self.subspace_b.space:
            raise _validation_error(
                "intersection_parent_mismatch",
                "subspaces must have the same field and axis",
            )
        return self


class GrassmannianCountRequest(StrictModel):
    field_order: int = Field(ge=2, le=MAX_FIELD_ORDER)
    ambient_dimension: int = Field(ge=1, le=MAX_DIM)
    subspace_dimension: int = Field(ge=0, le=MAX_DIM)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if not isprime(self.field_order):
            raise _validation_error(
                "field_order_not_prime", "field_order must be prime"
            )
        if self.subspace_dimension > self.ambient_dimension:
            raise _validation_error(
                "subspace_dimension_exceeds_ambient",
                "subspace dimension cannot exceed ambient dimension",
            )
        return self


class ProjectiveSpaceEnumerateRequest(StrictModel):
    """One finite projective space whose complete point list fits the envelope.

    ``q`` is the prime field order and ``n`` is the length of its ordered
    coordinate axis.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "One finite projective space whose complete canonical point "
                "list fits the transport envelope. It admits exactly q**n <= "
                f"{MAX_PROJECTIVE_SPACE_ENUMERATION_VECTORS}, where q is the "
                "prime field order and n is the length of its ordered coordinate "
                "axis; this leaves at most "
                f"{MAX_PROJECTIVE_SPACE_ENUMERATION_VECTORS - 1} canonical "
                "projective points, returned as a typed point sequence holding "
                "the parent space once plus bare coordinate tuples. Admission "
                "also predicts the complete serialized result before execution "
                "and rejects any request whose canonical output would exceed "
                f"the {MAX_PROJECTIVE_ENUMERATION_RESULT_BYTES}-byte result "
                "budget, so every accepted request returns its declared "
                "result inside the canonical transport limit."
            )
        }
    )

    space: PrimeFieldVectorSpace = Field(
        description=(
            "An ordered coordinate space over the prime field F_q. Complete "
            "enumeration requires q**len(axis) <= "
            f"{MAX_PROJECTIVE_SPACE_ENUMERATION_VECTORS} plus a predicted "
            f"serialized result within the "
            f"{MAX_PROJECTIVE_ENUMERATION_RESULT_BYTES}-byte budget; the "
            "canonical sequence holds at most "
            f"{MAX_PROJECTIVE_SPACE_ENUMERATION_VECTORS - 1} bare coordinate "
            "tuples in this axis order."
        )
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        q = self.space.field_order
        n = len(self.space.axis)
        if q**n > MAX_PROJECTIVE_SPACE_ENUMERATION_VECTORS:
            raise _validation_error(
                "projective_space_too_large",
                "projective space exceeds the "
                f"{MAX_PROJECTIVE_SPACE_ENUMERATION_VECTORS}-vector "
                "enumeration envelope",
            )
        self.require_transportable_result()
        return self

    def require_transportable_result(self) -> None:
        """Reject requests whose complete canonical reply cannot fit transport.

        Runs before execution on admitted mathematics alone. Each returned
        point is a bare coordinate array with at most ``n`` entries carrying
        at most ``len(str(q - 1))`` digits, so the exact worst-case point
        array size follows from q and n; the parent space echoes once with
        NFC-normalized labels, matching how canonical JSON serialization
        normalizes string values before transport; and a fixed constant
        covers keys, method string, count digits, and punctuation.
        """

        q = self.space.field_order
        n = len(self.space.axis)
        digit_width = len(str(q - 1))
        point_count = (q**n - 1) // (q - 1)
        per_point_bytes = 2 + n * digit_width + (n - 1) + 1
        predicted = (
            _PROJECTIVE_ENUMERATION_ENVELOPE_BYTES
            + sum(
                len(encode_strict_json(unicodedata.normalize("NFC", label)))
                for label in self.space.axis
            )
            + point_count * per_point_bytes
        )
        if predicted > MAX_PROJECTIVE_ENUMERATION_RESULT_BYTES:
            raise _validation_error(
                "projective_enumeration_result_too_large",
                "the complete serialized point list would exceed the "
                f"{MAX_PROJECTIVE_ENUMERATION_RESULT_BYTES}-byte result budget",
            )


class ProjectivePointCanonicalizeResult(ProjectivePointCanonicalizeRequest):
    point: ProjectivePoint
    scale: int = Field(ge=1)
    method: str = "FIRST_NONZERO_TO_ONE"

    @model_validator(mode="after")
    def bind_point_parent(self) -> Self:
        if self.point.space != self.space:
            raise _validation_error(
                "canonicalization_parent_mismatch",
                "projective point must use the declared parent space",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: ProjectivePointCanonicalizeRequest,
        point: ProjectivePoint,
        scale: int,
    ) -> Self:
        return cls(**request.model_dump(), point=point, scale=scale)


class ProjectivePointEqualResult(StrictModel):
    point_a: ProjectivePoint
    point_b: ProjectivePoint
    equal: bool
    method: str = "CANONICAL_REPRESENTATIVE"

    @model_validator(mode="after")
    def require_same_parent(self) -> Self:
        if self.point_a.space != self.point_b.space:
            raise _validation_error(
                "projective_parent_mismatch",
                "projective points must have the same field and axis",
            )
        return self

    @classmethod
    def _from_kernel(cls, request: ProjectivePointEqualRequest, equal: bool) -> Self:
        return cls(point_a=request.point_a, point_b=request.point_b, equal=equal)


class SubspaceComputeResult(SubspaceComputeRequest):
    subspace: LinearSubspace
    method: str = "RREF"

    @model_validator(mode="after")
    def bind_subspace_parent(self) -> Self:
        if self.subspace.space != self.space:
            raise _validation_error(
                "subspace_parent_mismatch",
                "subspace must use the declared parent space",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: SubspaceComputeRequest, subspace: LinearSubspace
    ) -> Self:
        return cls(**request.model_dump(), subspace=subspace)


class SubspaceMembershipResult(StrictModel):
    subspace: LinearSubspace
    vector: tuple[int, ...]
    is_member: bool
    method: str = "RREF_MEMBERSHIP"

    @model_validator(mode="after")
    def bind_vector_parent(self) -> Self:
        _validate_vector(self.vector, self.subspace.space)
        return self

    @classmethod
    def _from_kernel(cls, request: SubspaceMembershipRequest, is_member: bool) -> Self:
        return cls(
            subspace=request.subspace, vector=request.vector, is_member=is_member
        )


class SubspaceSpanResult(SubspaceSpanRequest):
    subspace: LinearSubspace
    method: str = "RREF"

    @model_validator(mode="after")
    def bind_subspace_parent(self) -> Self:
        if self.subspace.space != self.space:
            raise _validation_error(
                "span_parent_mismatch", "subspace must use the declared parent space"
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: SubspaceSpanRequest, subspace: LinearSubspace
    ) -> Self:
        return cls(**request.model_dump(), subspace=subspace)


class SubspaceIntersectionResult(SubspaceIntersectionRequest):
    subspace: LinearSubspace
    method: str = "INTERSECTION"

    @model_validator(mode="after")
    def bind_subspace_parent(self) -> Self:
        if self.subspace.space != self.subspace_a.space:
            raise _validation_error(
                "intersection_result_parent_mismatch",
                "intersection must use the input parent space",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: SubspaceIntersectionRequest, subspace: LinearSubspace
    ) -> Self:
        return cls(**request.model_dump(), subspace=subspace)


class GrassmannianCountResult(StrictModel):
    field_order: int
    ambient_dimension: int
    subspace_dimension: int
    count: CanonicalInteger = Field(
        description="Exact Gaussian-binomial count encoded as a canonical decimal integer."
    )
    method: str = "GAUSSIAN_BINOMIAL"

    @model_validator(mode="after")
    def require_parameter_domain(self) -> Self:
        if not isprime(self.field_order) or not (
            0 <= self.subspace_dimension <= self.ambient_dimension <= MAX_DIM
        ):
            raise _validation_error(
                "grassmannian_parameters_invalid",
                "Grassmannian parameters are outside the public domain",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: GrassmannianCountRequest, count: CanonicalInteger
    ) -> Self:
        return cls(
            field_order=request.field_order,
            ambient_dimension=request.ambient_dimension,
            subspace_dimension=request.subspace_dimension,
            count=count,
        )


class ProjectiveSpaceEnumerateResult(StrictModel):
    """The complete canonical point sequence of one finite projective space.

    The sequence value owns its declared parent space and self-certifies
    completeness, normalization, and uniqueness; natively it iterates typed
    :class:`ProjectivePoint` items while serializing as the parent space once
    plus one bare coordinate tuple per point.
    """

    sequence: ProjectivePointSequence
    method: str = "CANONICAL_REPRESENTATIVES"


# ---------------------------------------------------------------------------
# Affine plane AG(2, q) over a prime field
# ---------------------------------------------------------------------------

#: Maximum prime-field order for which the complete affine plane fits the
#: reused :class:`IncidenceStructure` transport budget.  ``q = 7`` yields
#: 49 points and 56 lines, both within the ``IncidenceStructure`` caps of
#: 100 points and 100 blocks; q = 11 would exceed both.
MAX_AFFINE_PLANE_FIELD_ORDER = 7


class ParallelClass(StrictModel):
    """One ordered parallel class of affine-plane line indices.

    ``line_ids`` are indices into the ``block_ids`` axis of the enclosing
    :class:`IncidenceStructure`.  The canonical ordering is slope-class-first
    then intercept, matching the line enumeration order.
    """

    line_ids: tuple[int, ...] = Field(min_length=1)
    label: str = Field(default="", max_length=64)


class PrimeFieldAffinePlaneRequest(StrictModel):
    """Construct the complete affine plane AG(2, q) over a prime field.

    ``prime_order`` is the prime field order q.  The result holds q^2 labelled
    points, q(q+1) labelled affine lines, exact point-line incidences, and a
    partition of the line axis into q+1 parallel classes.

    The admitted domain is ``2 <= q <= MAX_AFFINE_PLANE_FIELD_ORDER``; this
    bound is dictated by the reused :class:`IncidenceStructure` transport
    budget (at most 100 points and 100 blocks), not by the mathematical
    construction, which is valid for every prime q.
    """

    prime_order: int = Field(ge=2, le=MAX_AFFINE_PLANE_FIELD_ORDER)

    @model_validator(mode="after")
    def require_prime(self) -> Self:
        if not isprime(self.prime_order):
            raise _validation_error(
                "affine_plane_order_not_prime", "prime_order must be prime"
            )
        return self


class PrimeFieldAffinePlaneResult(StrictModel):
    """The complete prime-field affine plane AG(2, q).

    The incidence structure reuses :class:`IncidenceStructure` with
    ``points`` and ``block_ids``/``blocks`` in a stable enumeration order:
    points (x, y) in lexicographic order (index = x * q + y), lines
    L_{m,b} = {(x, mx+b mod q) : x in F_q} for each slope m and intercept b,
    then vertical lines V_b = {(b, y) : y in F_q} for each b.
    ``parallel_classes`` partitions the line axis into q+1 ordered classes.
    """

    prime_order: int = Field(ge=2, le=MAX_AFFINE_PLANE_FIELD_ORDER)
    incidence: IncidenceStructure
    parallel_classes: tuple[ParallelClass, ...] = Field(min_length=2)
    total_incidences: int = Field(ge=0)
    method: str = "MODULAR_ARITHMETIC"

    @model_validator(mode="after")
    def require_consistency(self) -> Self:
        q = self.prime_order
        if not isprime(q):
            raise _validation_error(
                "affine_plane_result_not_prime", "prime_order must be prime"
            )
        expected_points = q * q
        if len(self.incidence.points) != expected_points:
            raise _validation_error(
                "affine_plane_point_count_mismatch",
                "affine plane must have exactly q^2 points",
            )
        expected_blocks = q * (q + 1)
        if len(self.incidence.block_ids) != expected_blocks:
            raise _validation_error(
                "affine_plane_block_count_mismatch",
                "affine plane must have exactly q(q+1) lines",
            )
        if len(self.parallel_classes) != q + 1:
            raise _validation_error(
                "affine_plane_class_count_mismatch",
                "affine plane must have exactly q+1 parallel classes",
            )
        flat_line_ids: list[int] = []
        for cls_ in self.parallel_classes:
            flat_line_ids.extend(cls_.line_ids)
        if flat_line_ids != list(range(expected_blocks)):
            raise _validation_error(
                "affine_plane_partition_mismatch",
                "parallel classes must partition line IDs in order",
            )
        expected_incidences = expected_points * (q + 1)
        if self.total_incidences != expected_incidences:
            raise _validation_error(
                "affine_plane_incidence_count_mismatch",
                "total_incidences must equal q^2 * (q+1)",
            )
        return self
