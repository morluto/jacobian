"""Native operations on canonical finite vector-family values."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math._rational_height import RationalHeight, sum_heights
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.number_theory.number_fields import GaussianRational
from jacobian.math.number_theory.number_fields.values import (
    MAX_GAUSSIAN_RATIONAL_COMPONENT_DIGITS,
)
from jacobian.math.topology.frames._flint import integer_gram, integer_gram_and_rank
from jacobian.math.topology.frames._models import (
    CoherenceResult,
    ComplexFrameProfileRequest,
    ComplexFrameProfileResult,
    FramePotentialResult,
    GramResult,
    MutuallyUnbiasedBasesRequest,
    MutuallyUnbiasedBasesResult,
    SicProfileRequest,
    SicProfileResult,
    TightEquiangularProfileResult,
)
from jacobian.math.topology.frames.values import (
    MAX_COMPLEX_BASIS_PAIRS,
    MAX_COMPLEX_COMPONENT_DIGITS,
    MAX_COMPLEX_INNER_PRODUCT_WORK,
    MAX_COMPLEX_PROFILE_CELLS,
    ComplexFrame,
    VectorFamily,
)

__all__ = [
    "coherence",
    "complex_frame_profile",
    "frame_potential",
    "gram",
    "mutually_unbiased_bases",
    "sic_profile",
    "tight_equiangular_profile",
    "verify_gram",
]


MAX_FRAME_GRAM_ENTRIES = 2_097_152
MAX_FRAME_GRAM_MULTIPLY_ADDS = 536_870_912


def _complex_parts(value: GaussianRational) -> tuple[Fraction, Fraction]:
    return value.as_fractions()


def _inner_product(
    left: tuple[GaussianRational, ...], right: tuple[GaussianRational, ...]
) -> tuple[Fraction, Fraction]:
    real = Fraction(0)
    imaginary = Fraction(0)
    for first, second in zip(left, right, strict=True):
        first_real, first_imaginary = _complex_parts(first)
        second_real, second_imaginary = _complex_parts(second)
        real += first_real * second_real + first_imaginary * second_imaginary
        imaginary += first_real * second_imaginary - first_imaginary * second_real
    return real, imaginary


def _norm_squared(vector: tuple[GaussianRational, ...]) -> Fraction:
    real, imaginary = _inner_product(vector, vector)
    if imaginary:
        raise ValueError("complex Hermitian norm unexpectedly has an imaginary part")
    return real


def _maximum_component_height(frame: ComplexFrame) -> RationalHeight:
    heights = tuple(
        RationalHeight.from_canonical(component)
        for vector in frame.vectors
        for scalar in vector
        for component in (scalar.real, scalar.imaginary)
    )
    if not heights:
        return RationalHeight(1, 1)
    return RationalHeight(
        max(height.numerator_digits for height in heights),
        max(height.denominator_digits for height in heights),
    )


def _complex_product_height(source: RationalHeight) -> RationalHeight:
    product = source.product(source)
    # Both real and imaginary parts are sums of two rational products.
    return sum_heights((product, product))


def _complex_sum_height(term: RationalHeight, count: int) -> RationalHeight:
    return sum_heights((term,) * count)


def _require_complex_accumulation_height(
    frame: ComplexFrame,
    *,
    normalized_operator: bool,
    normalized_overlaps: bool,
) -> None:
    """Admit rational growth before constructing any exact arithmetic values."""

    source = _maximum_component_height(frame)
    product = _complex_product_height(source)
    norm = _complex_sum_height(product, 2 * frame.dimension)

    operator_term = product
    if normalized_operator:
        operator_term = product.quotient(norm)
    operator = _complex_sum_height(operator_term, len(frame.vectors))
    trace = _complex_sum_height(operator, frame.dimension).quotient(
        RationalHeight(1, max(1, len(str(frame.dimension))))
    )
    residual = sum_heights((operator, trace))
    if (
        max(
            operator.numerator_digits,
            operator.denominator_digits,
            residual.numerator_digits,
            residual.denominator_digits,
        )
        > MAX_GAUSSIAN_RATIONAL_COMPONENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("frame", "vectors"),
            code="frames.complex_scalar_height",
            message="Gaussian-rational frame accumulation exceeds its admitted height",
        )

    if normalized_overlaps:
        inner_product = _complex_sum_height(product, frame.dimension)
        squared_magnitude = sum_heights((inner_product.product(inner_product),) * 2)
        denominator = norm.product(norm)
        overlap = squared_magnitude.quotient(denominator)
        if overlap.exceeds(MAX_CANONICAL_RATIONAL_DIGITS):
            raise OperationResourceAdmissionError(
                location=("frame", "vectors"),
                code="frames.complex_overlap_height",
                message="normalized complex overlap exceeds its admitted height",
            )


def _complex_frame_admitted(
    frame: ComplexFrame,
    *,
    normalized_operator: bool = False,
    normalized_overlaps: bool = False,
) -> tuple[Fraction, ...]:
    if len(frame.vectors) * frame.dimension > MAX_COMPLEX_PROFILE_CELLS:
        raise OperationResourceAdmissionError(
            location=("frame", "vectors"),
            code="frames.complex_profile_cells",
            message="complex profile cells exceed the admitted output envelope",
        )
    if any(
        max(
            len(str(abs(component.num))),
            len(str(component.den)),
        )
        > MAX_COMPLEX_COMPONENT_DIGITS
        for vector in frame.vectors
        for scalar in vector
        for component in (scalar.real, scalar.imaginary)
    ):
        raise OperationResourceAdmissionError(
            location=("frame", "vectors"),
            code="frames.complex_scalar_digits",
            message="complex scalar components exceed the admitted profile height",
        )
    _require_complex_accumulation_height(
        frame,
        normalized_operator=normalized_operator,
        normalized_overlaps=normalized_overlaps,
    )
    norms = tuple(_norm_squared(vector) for vector in frame.vectors)
    if any(norm <= 0 for norm in norms):
        raise OperationDomainValidationError(
            location=("frame", "vectors"),
            code="frames.zero_complex_vector",
            message="complex frame profiles require every vector to be nonzero",
        )
    return norms


def _require_complex_profile_work(
    frame: ComplexFrame, *, emitted_matrix_cells: int
) -> None:
    vectors = len(frame.vectors)
    pair_cells = vectors * vectors
    work = pair_cells * frame.dimension + len(frame.vectors) * frame.dimension**2
    if emitted_matrix_cells > MAX_COMPLEX_PROFILE_CELLS:
        raise OperationResourceAdmissionError(
            location=("frame",),
            code="frames.complex_profile_output",
            message="complex profile output cells exceed the admitted envelope",
        )
    if work > MAX_COMPLEX_INNER_PRODUCT_WORK:
        raise OperationResourceAdmissionError(
            location=("frame",),
            code="frames.complex_profile_work",
            message="complex profile inner-product work exceeds the admitted envelope",
        )


def _tight_complex_frame(
    frame: ComplexFrame,
    *,
    normalized_projectors: bool = False,
    norms: tuple[Fraction, ...] | None = None,
) -> tuple[
    bool,
    tuple[tuple[GaussianRational, ...], ...],
    tuple[tuple[GaussianRational, ...], ...],
]:
    if normalized_projectors and norms is None:
        raise ValueError("normalized frame operators require source norms")
    operator: list[list[tuple[Fraction, Fraction]]] = [
        [(Fraction(0), Fraction(0)) for _ in range(frame.dimension)]
        for row in range(frame.dimension)
    ]
    for index, vector in enumerate(frame.vectors):
        scale = Fraction(1)
        if normalized_projectors:
            assert norms is not None
            scale = Fraction(1, 1) / norms[index]
        for row in range(frame.dimension):
            first_real, first_imaginary = _complex_parts(vector[row])
            for column in range(frame.dimension):
                second_real, second_imaginary = _complex_parts(vector[column])
                real = first_real * second_real + first_imaginary * second_imaginary
                imaginary = (
                    first_imaginary * second_real - first_real * second_imaginary
                )
                previous_real, previous_imaginary = operator[row][column]
                operator[row][column] = (
                    previous_real + scale * real,
                    previous_imaginary + scale * imaginary,
                )
    trace: Fraction = sum(
        (operator[index][index][0] for index in range(frame.dimension)), Fraction(0)
    )
    scalar: Fraction = trace * Fraction(1, frame.dimension)
    residual: list[list[tuple[Fraction, Fraction]]] = [
        [
            (
                operator[row][column][0] - (scalar if row == column else Fraction(0)),
                operator[row][column][1] - Fraction(0),
            )
            for column in range(frame.dimension)
        ]
        for row in range(frame.dimension)
    ]

    def to_value(pair: tuple[Fraction, Fraction]) -> GaussianRational:
        return GaussianRational.from_fractions(*pair)

    operator_value = tuple(tuple(to_value(entry) for entry in row) for row in operator)
    residual_value = tuple(tuple(to_value(entry) for entry in row) for row in residual)
    return (
        all(entry == (Fraction(0), Fraction(0)) for row in residual for entry in row),
        operator_value,
        residual_value,
    )


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


def complex_frame_profile(
    request: ComplexFrameProfileRequest,
) -> ComplexFrameProfileResult:
    """Return an exact complex frame operator and tight/equiangular profile."""

    dimension = request.frame.dimension
    _require_complex_profile_work(
        request.frame, emitted_matrix_cells=2 * dimension * dimension
    )
    norms = _complex_frame_admitted(request.frame, normalized_overlaps=True)
    equiangular, common = _equiangular_complex_frame(request.frame, norms)
    tight, frame_operator, tight_residual = _tight_complex_frame(request.frame)
    return ComplexFrameProfileResult._from_kernel(
        request,
        tight=tight,
        equiangular=equiangular,
        common_squared_overlap=(
            CanonicalRational.from_fraction(common)
            if equiangular and common is not None
            else None
        ),
        frame_operator=frame_operator,
        tight_residual=tight_residual,
    )


def mutually_unbiased_bases(
    request: MutuallyUnbiasedBasesRequest,
) -> MutuallyUnbiasedBasesResult:
    """Decide exact mutual unbiasedness of a bounded complex basis family."""

    dimension = request.dimension
    basis_count = len(request.bases)
    pair_count = basis_count * (basis_count - 1) // 2
    work = (pair_count + basis_count) * dimension * dimension * dimension
    output_cells = (basis_count + pair_count) * dimension * dimension
    if pair_count > MAX_COMPLEX_BASIS_PAIRS or work > MAX_COMPLEX_INNER_PRODUCT_WORK:
        raise OperationResourceAdmissionError(
            location=("bases",),
            code="frames.mub_work",
            message="MUB basis-pair work exceeds the admitted envelope",
        )
    if output_cells > MAX_COMPLEX_PROFILE_CELLS:
        raise OperationResourceAdmissionError(
            location=("bases",),
            code="frames.mub_output",
            message="MUB ledger output exceeds the admitted envelope",
        )
    basis_norms = tuple(
        _complex_frame_admitted(basis, normalized_overlaps=True)
        for basis in request.bases
    )
    unbiased = True
    basis_grams: list[tuple[tuple[GaussianRational, ...], ...]] = []
    for basis in request.bases:
        basis_grams.append(
            tuple(
                tuple(
                    GaussianRational.from_fractions(*_inner_product(left, right))
                    for right in basis.vectors
                )
                for left in basis.vectors
            )
        )
    cross_gram_squared: list[tuple[tuple[CanonicalRational, ...], ...]] = []
    for basis, norms in zip(request.bases, basis_norms, strict=True):
        for left_index in range(dimension):
            for right_index in range(left_index + 1, dimension):
                if _inner_product(
                    basis.vectors[left_index], basis.vectors[right_index]
                ) != (Fraction(0), Fraction(0)):
                    unbiased = False
        if any(norm <= 0 for norm in norms):
            unbiased = False
    for first in range(len(request.bases)):
        for second in range(first + 1, len(request.bases)):
            cross: list[tuple[CanonicalRational, ...]] = []
            for left_vector, left_norm in zip(
                request.bases[first].vectors, basis_norms[first], strict=True
            ):
                row: list[CanonicalRational] = []
                for right_vector, right_norm in zip(
                    request.bases[second].vectors, basis_norms[second], strict=True
                ):
                    real, imaginary = _inner_product(left_vector, right_vector)
                    if (
                        real * real + imaginary * imaginary
                    ) * dimension != left_norm * right_norm:
                        unbiased = False
                    row.append(
                        CanonicalRational.from_fraction(
                            (real * real + imaginary * imaginary)
                            / (left_norm * right_norm)
                        )
                    )
                cross.append(tuple(row))
            cross_gram_squared.append(tuple(cross))
    return MutuallyUnbiasedBasesResult._from_kernel(
        request,
        basis_grams=tuple(basis_grams),
        cross_gram_squared=tuple(cross_gram_squared),
        is_mutually_unbiased=unbiased,
    )


def sic_profile(request: SicProfileRequest) -> SicProfileResult:
    """Decide the exact SIC overlap equations for a complex vector family."""

    frame = request.frame
    _require_complex_profile_work(
        frame,
        emitted_matrix_cells=len(frame.vectors) ** 2
        + 2 * frame.dimension * frame.dimension,
    )
    norms = _complex_frame_admitted(
        frame, normalized_operator=True, normalized_overlaps=True
    )
    expected_count = frame.dimension * frame.dimension
    is_sic = len(frame.vectors) == expected_count
    squared_overlaps: list[tuple[CanonicalRational, ...]] = []
    for left_vector, left_norm in zip(frame.vectors, norms, strict=True):
        row: list[CanonicalRational] = []
        for right_vector, right_norm in zip(frame.vectors, norms, strict=True):
            real, imaginary = _inner_product(left_vector, right_vector)
            row.append(
                CanonicalRational.from_fraction(
                    (real * real + imaginary * imaginary) / (left_norm * right_norm)
                )
            )
        squared_overlaps.append(tuple(row))
    common: Fraction | None = None
    if is_sic:
        common = Fraction(1, frame.dimension + 1)
        for left_index in range(len(frame.vectors)):
            for right_index in range(left_index + 1, len(frame.vectors)):
                real, imaginary = _inner_product(
                    frame.vectors[left_index], frame.vectors[right_index]
                )
                overlap = (real * real + imaginary * imaginary) / (
                    norms[left_index] * norms[right_index]
                )
                if overlap != common:
                    is_sic = False
    tight, frame_operator, tight_residual = _tight_complex_frame(
        frame, normalized_projectors=True, norms=norms
    )
    is_sic = is_sic and tight
    return SicProfileResult._from_kernel(
        request,
        is_sic=is_sic,
        common_squared_overlap=(
            CanonicalRational.from_fraction(Fraction(1, frame.dimension + 1))
            if is_sic
            else None
        ),
        squared_overlaps=tuple(squared_overlaps),
        frame_operator=frame_operator,
        tight_residual=tight_residual,
    )
