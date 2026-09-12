"""Native operations on canonical finite vector-family values."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.topology.frames._flint import integer_gram, integer_gram_and_rank
from jacobian.math.topology.frames._models import (
    CoherenceResult,
    ComplexFrameDesignProfileRequest,
    ComplexFrameDesignProfileResult,
    FramePotentialResult,
    GramResult,
    MutuallyUnbiasedBasesRequest,
    MutuallyUnbiasedBasesResult,
    SicProfileRequest,
    SicProfileResult,
    TightEquiangularProfileResult,
)
from jacobian.math.topology.frames.values import (
    ComplexFrame,
    ExactComplex,
    VectorFamily,
)

__all__ = [
    "coherence",
    "complex_design_profile",
    "frame_potential",
    "gram",
    "mutually_unbiased_bases",
    "sic_profile",
    "tight_equiangular_profile",
    "verify_gram",
]


MAX_FRAME_GRAM_ENTRIES = 2_097_152
MAX_FRAME_GRAM_MULTIPLY_ADDS = 536_870_912


def _complex_parts(value: ExactComplex) -> tuple[Fraction, Fraction]:
    return value.real.as_fraction(), value.imaginary.as_fraction()


def _inner_product(
    left: tuple[ExactComplex, ...], right: tuple[ExactComplex, ...]
) -> tuple[Fraction, Fraction]:
    real = Fraction(0)
    imaginary = Fraction(0)
    for first, second in zip(left, right, strict=True):
        first_real, first_imaginary = _complex_parts(first)
        second_real, second_imaginary = _complex_parts(second)
        real += first_real * second_real + first_imaginary * second_imaginary
        imaginary += first_real * second_imaginary - first_imaginary * second_real
    return real, imaginary


def _norm_squared(vector: tuple[ExactComplex, ...]) -> Fraction:
    real, imaginary = _inner_product(vector, vector)
    if imaginary:
        raise ValueError("complex Hermitian norm unexpectedly has an imaginary part")
    return real


def _complex_frame_admitted(frame: ComplexFrame) -> tuple[Fraction, ...]:
    norms = tuple(_norm_squared(vector) for vector in frame.vectors)
    if any(norm <= 0 for norm in norms):
        raise OperationDomainValidationError(
            location=("frame", "vectors"),
            code="frames.zero_complex_vector",
            message="complex frame profiles require every vector to be nonzero",
        )
    return norms


def _tight_complex_frame(frame: ComplexFrame) -> bool:
    operator = [
        [
            (
                sum(
                    _inner_product((vector[row],), (vector[column],))[0]
                    for vector in frame.vectors
                ),
                sum(
                    _inner_product((vector[row],), (vector[column],))[1]
                    for vector in frame.vectors
                ),
            )
            for column in range(frame.dimension)
        ]
        for row in range(frame.dimension)
    ]
    scalar = operator[0][0]
    return all(
        operator[row][column] == (scalar if row == column else (Fraction(0), Fraction(0)))
        for row in range(frame.dimension)
        for column in range(frame.dimension)
    ) and scalar[1] == 0


def _equiangular_complex_frame(
    frame: ComplexFrame, norms: tuple[Fraction, ...]
) -> tuple[bool, Fraction | None]:
    if not norms or any(norm != norms[0] for norm in norms):
        return False, None
    common: Fraction | None = None
    for left in range(len(frame.vectors)):
        for right in range(left + 1, len(frame.vectors)):
            real, imaginary = _inner_product(frame.vectors[left], frame.vectors[right])
            candidate = (real * real + imaginary * imaginary) / (
                norms[left] * norms[right]
            )
            if common is None:
                common = candidate
            elif candidate != common:
                return False, None
    return True, common


def _require_gram_work_budget(value: VectorFamily) -> None:
    vector_count = len(value.vectors)
    dimension = value.dimension
    gram_entries = vector_count**2
    if gram_entries > MAX_FRAME_GRAM_ENTRIES:
        raise OperationResourceAdmissionError(
            location=("vectors",),
            code="frames.gram_intermediate_budget",
            message="frame Gram intermediate exceeds its entry budget",
        )
    if gram_entries * dimension > MAX_FRAME_GRAM_MULTIPLY_ADDS:
        raise OperationResourceAdmissionError(
            location=("vectors",),
            code="frames.gram_work_budget",
            message="frame Gram computation exceeds its multiply-add work budget",
        )


def _gram_result(value: VectorFamily) -> GramResult:
    matrix = integer_gram(value.vectors)
    return GramResult._from_kernel(
        vectors=value.vectors,
        dimension=value.dimension,
        gram=IntegerMatrix(
            row_count=len(matrix),
            column_count=len(matrix[0]) if matrix else len(value.vectors),
            entries=matrix,
        ),
    )


def _admit_frame(value: VectorFamily, *, rank: int) -> None:
    if rank != value.dimension:
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.frame_does_not_span",
            message="a finite frame must span its ambient space",
        )


def _admit_frame_shape(value: VectorFamily) -> None:
    if len(value.vectors) < value.dimension:
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.frame_does_not_span",
            message="a finite frame must have at least as many vectors as coordinates",
        )


def gram(value: VectorFamily) -> GramResult:
    """Compute the exact Gram matrix of a vector family."""
    _require_gram_work_budget(value)
    return _gram_result(value)


def verify_gram(claim: GramResult) -> bool:
    """Verify the retained vector-family Gram relation for a decoded claim."""
    try:
        if (
            claim.gram.row_count != len(claim.vectors)
            or claim.gram.column_count != len(claim.vectors)
            or any(len(vector) != claim.dimension for vector in claim.vectors)
        ):
            return False
    except (AttributeError, IndexError, TypeError):
        return False
    _require_gram_work_budget(claim)
    return claim.gram.entries == integer_gram(claim.vectors)


def coherence(value: VectorFamily) -> CoherenceResult:
    """Compute exact normalized squared coherence of a finite frame."""
    _require_gram_work_budget(value)
    if any(not any(vector) for vector in value.vectors):
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.zero_vector",
            message="coherence requires every vector to be nonzero",
        )
    _admit_frame_shape(value)
    rank, matrix = integer_gram_and_rank(value.vectors, dimension=value.dimension)
    _admit_frame(value, rank=rank)
    assert matrix is not None
    maximum_numerator = 0
    maximum_denominator = 1
    pair: tuple[int, int] | None = None
    for left in range(len(value.vectors)):
        for right in range(left + 1, len(value.vectors)):
            inner_product = matrix[left][right]
            denominator = matrix[left][left] * matrix[right][right]
            numerator = inner_product * inner_product
            candidate_pair = (left, right)
            comparison = numerator * maximum_denominator - (
                maximum_numerator * denominator
            )
            if comparison > 0 or (
                comparison == 0 and (pair is None or candidate_pair > pair)
            ):
                maximum_numerator = numerator
                maximum_denominator = denominator
                pair = candidate_pair
    return CoherenceResult._from_kernel(
        vectors=value.vectors,
        dimension=value.dimension,
        coherence_squared=CanonicalRational.from_fraction(
            Fraction(maximum_numerator, maximum_denominator)
        ),
        maximizing_pair=pair,
    )


def frame_potential(value: VectorFamily) -> FramePotentialResult:
    """Compute the exact frame potential of a finite frame."""
    _require_gram_work_budget(value)
    _admit_frame_shape(value)
    rank, matrix = integer_gram_and_rank(value.vectors, dimension=value.dimension)
    _admit_frame(value, rank=rank)
    assert matrix is not None
    total = sum(entry**2 for row in matrix for entry in row)
    return FramePotentialResult._from_kernel(
        vectors=value.vectors, dimension=value.dimension, potential=total
    )


def tight_equiangular_profile(value: VectorFamily) -> TightEquiangularProfileResult:
    """Classify tightness and equiangularity using exact integer Gram data."""

    _require_gram_work_budget(value)
    if any(not any(vector) for vector in value.vectors):
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.zero_vector",
            message="frame profiles require every vector to be nonzero",
        )
    _admit_frame_shape(value)
    rank, matrix = integer_gram_and_rank(value.vectors, dimension=value.dimension)
    _admit_frame(value, rank=rank)
    assert matrix is not None
    dimension = value.dimension
    frame_operator = [
        [
            sum(vector[row] * vector[column] for vector in value.vectors)
            for column in range(dimension)
        ]
        for row in range(dimension)
    ]
    diagonal = frame_operator[0][0] if dimension else 0
    tight = all(
        frame_operator[row][column] == (diagonal if row == column else 0)
        for row in range(dimension)
        for column in range(dimension)
    )
    common: Fraction | None = None
    equiangular = True
    for left in range(len(value.vectors)):
        for right in range(left + 1, len(value.vectors)):
            candidate = Fraction(
                matrix[left][right] ** 2,
                matrix[left][left] * matrix[right][right],
            )
            if common is None:
                common = candidate
            elif candidate != common:
                equiangular = False
    return TightEquiangularProfileResult._from_kernel(
        vectors=value.vectors,
        dimension=value.dimension,
        tight=tight,
        tight_constant=diagonal if tight else None,
        equiangular=equiangular,
        common_squared_inner_product=(
            CanonicalRational.from_fraction(common)
            if equiangular and common is not None
            else None
        ),
    )


def complex_design_profile(
    request: ComplexFrameDesignProfileRequest,
) -> ComplexFrameDesignProfileResult:
    """Return exact tightness and equal-norm equiangularity of a complex frame."""

    norms = _complex_frame_admitted(request.frame)
    equiangular, common = _equiangular_complex_frame(request.frame, norms)
    return ComplexFrameDesignProfileResult._from_kernel(
        request,
        tight=_tight_complex_frame(request.frame),
        equiangular=equiangular,
        common_squared_overlap=(
            CanonicalRational.from_fraction(common) if equiangular and common is not None else None
        ),
    )


def mutually_unbiased_bases(
    request: MutuallyUnbiasedBasesRequest,
) -> MutuallyUnbiasedBasesResult:
    """Decide exact mutual unbiasedness of a bounded complex basis family."""

    basis_norms = tuple(_complex_frame_admitted(basis) for basis in request.bases)
    unbiased = True
    dimension = request.dimension
    for basis, norms in zip(request.bases, basis_norms, strict=True):
        for left in range(dimension):
            for right in range(left + 1, dimension):
                if _inner_product(basis.vectors[left], basis.vectors[right]) != (0, 0):
                    unbiased = False
        if any(norm <= 0 for norm in norms):
            unbiased = False
    for first in range(len(request.bases)):
        for second in range(first + 1, len(request.bases)):
            for left, left_norm in zip(
                request.bases[first].vectors, basis_norms[first], strict=True
            ):
                for right, right_norm in zip(
                    request.bases[second].vectors, basis_norms[second], strict=True
                ):
                    real, imaginary = _inner_product(left, right)
                    if (
                        real * real + imaginary * imaginary
                    ) * dimension != left_norm * right_norm:
                        unbiased = False
    return MutuallyUnbiasedBasesResult._from_kernel(
        request, is_mutually_unbiased=unbiased
    )


def sic_profile(request: SicProfileRequest) -> SicProfileResult:
    """Decide the exact SIC overlap equations for a complex vector family."""

    frame = request.frame
    norms = _complex_frame_admitted(frame)
    expected_count = frame.dimension * frame.dimension
    is_sic = len(frame.vectors) == expected_count
    common: Fraction | None = None
    if is_sic:
        is_sic = all(norm == norms[0] for norm in norms)
        common = Fraction(1, frame.dimension + 1)
        for left in range(len(frame.vectors)):
            for right in range(left + 1, len(frame.vectors)):
                real, imaginary = _inner_product(frame.vectors[left], frame.vectors[right])
                overlap = (real * real + imaginary * imaginary) / (
                    norms[left] * norms[right]
                )
                if overlap != common:
                    is_sic = False
    return SicProfileResult._from_kernel(
        request,
        is_sic=is_sic,
        common_squared_overlap=(
            CanonicalRational.from_fraction(Fraction(1, frame.dimension + 1))
            if is_sic
            else None
        ),
    )
