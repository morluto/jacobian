"""Native operations on canonical finite vector-family values."""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    canonical_rational_component_digits,
    require_bounded_rational,
)
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
    ComplexFrameProfileResult,
    FramePotentialResult,
    GramResult,
    MutuallyUnbiasedBasesResult,
    SicProfileResult,
    TightEquiangularProfileResult,
)
from jacobian.math.topology.frames.values import (
    _MAX_VECTOR_ENTRY,
    MAX_COMPLEX_BASIS_COUNT,
    MAX_COMPLEX_BASIS_PAIRS,
    MAX_COMPLEX_COMPONENT_DIGITS,
    MAX_COMPLEX_FRAME_CELLS,
    MAX_COMPLEX_INNER_PRODUCT_WORK,
    MAX_COMPLEX_PROFILE_CELLS,
    MAX_DIM,
    MAX_VECTOR_CELLS,
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


def _admit_canonical_component(
    component: object, *, location: tuple[str | int, ...]
) -> None:
    if type(component) is not CanonicalRational:
        raise OperationDomainValidationError(
            location=location,
            code="frames.complex_scalar_component_type",
            message="Gaussian-rational components must be canonical rationals",
        )
    assert isinstance(component, CanonicalRational)
    numerator = getattr(component, "num", None)
    denominator = getattr(component, "den", None)
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise OperationDomainValidationError(
            location=location,
            code="frames.complex_scalar_component",
            message="Gaussian-rational components must be canonical rationals",
        )
    try:
        require_bounded_rational(
            component,
            max_digits=MAX_GAUSSIAN_RATIONAL_COMPONENT_DIGITS,
            label="Gaussian-rational component",
        )
    except ValueError:
        raise OperationResourceAdmissionError(
            location=location,
            code="frames.complex_scalar_height",
            message="Gaussian-rational components exceed the admitted height",
        ) from None
    if (numerator == 0 and denominator != 1) or gcd(abs(numerator), denominator) != 1:
        raise OperationDomainValidationError(
            location=location,
            code="frames.complex_scalar_component",
            message="Gaussian-rational components must be canonical rationals",
        )


def _admit_complex_frame(
    frame: object, *, location: tuple[str | int, ...] = ("frame",)
) -> None:
    """Recheck the runtime shape of a complex frame at native boundaries.

    ``model_construct`` is intentionally available to trusted producers and
    therefore bypasses Pydantic validators.  Native callers can still provide
    such a value, so every operation that indexes complex coordinates must
    establish the frame's carrier, axes, and scalar shape before doing work.
    """

    if type(frame) is not ComplexFrame:
        raise OperationDomainValidationError(
            location=location,
            code="frames.complex_frame_type",
            message="complex frame must be a ComplexFrame value",
        )
    dimension = getattr(frame, "dimension", None)
    if type(dimension) is not int or not 1 <= dimension <= MAX_DIM:
        raise OperationDomainValidationError(
            location=(*location, "dimension"),
            code="frames.complex_frame_dimension",
            message="complex frame dimension must be a positive bounded integer",
        )
    vectors = getattr(frame, "vectors", None)
    if type(vectors) is not tuple:
        raise OperationDomainValidationError(
            location=(*location, "vectors"),
            code="frames.complex_frame_vectors",
            message="complex frame vectors must be an exact tuple",
        )
    if not vectors:
        raise OperationDomainValidationError(
            location=(*location, "vectors"),
            code="frames.empty_complex_frame",
            message="complex frame must contain at least one vector",
        )
    if len(vectors) * dimension > MAX_COMPLEX_FRAME_CELLS:
        raise OperationResourceAdmissionError(
            location=(*location, "vectors"),
            code="frames.complex_vector_cell_budget",
            message="complex vector family exceeds the materialized-cell budget",
        )
    for vector_index, vector in enumerate(vectors):
        vector_location = (*location, "vectors", vector_index)
        if type(vector) is not tuple:
            raise OperationDomainValidationError(
                location=vector_location,
                code="frames.complex_vector_type",
                message="complex frame vectors must be exact tuples",
            )
        if len(vector) != dimension:
            raise OperationDomainValidationError(
                location=vector_location,
                code="frames.complex_vector_dimension_mismatch",
                message="all complex vectors must have equal dimension",
            )
        for coordinate_index, scalar in enumerate(vector):
            scalar_location = (*vector_location, coordinate_index)
            if type(scalar) is not GaussianRational:
                raise OperationDomainValidationError(
                    location=scalar_location,
                    code="frames.complex_scalar_type",
                    message="complex coordinates must be GaussianRational values",
                )
            for component_name in ("real", "imaginary"):
                component = getattr(scalar, component_name, None)
                _admit_canonical_component(
                    component, location=(*scalar_location, component_name)
                )


def _admit_vector_family(
    value: object, *, expected_type: type[VectorFamily] = VectorFamily
) -> VectorFamily:
    """Admit a native vector-family value before integer-kernel work."""

    if type(value) is not expected_type:
        raise OperationDomainValidationError(
            location=(),
            code="frames.vector_family_type",
            message=f"operation requires a {expected_type.__name__} value",
        )
    dimension = getattr(value, "dimension", None)
    if type(dimension) is not int or not 0 <= dimension <= MAX_DIM:
        raise OperationDomainValidationError(
            location=("dimension",),
            code="frames.vector_family_dimension",
            message="vector-family dimension must be a bounded nonnegative integer",
        )
    vectors = getattr(value, "vectors", None)
    if type(vectors) is not tuple:
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.vector_family_vectors",
            message="vector-family vectors must be an exact tuple",
        )
    if len(vectors) * dimension > MAX_VECTOR_CELLS:
        raise OperationResourceAdmissionError(
            location=("vectors",),
            code="frames.vector_cell_budget",
            message="vector family exceeds the materialized-cell budget",
        )
    for vector_index, vector in enumerate(vectors):
        vector_location = ("vectors", vector_index)
        if type(vector) is not tuple:
            raise OperationDomainValidationError(
                location=vector_location,
                code="frames.vector_type",
                message="vectors must be exact tuples",
            )
        if len(vector) != dimension:
            raise OperationDomainValidationError(
                location=vector_location,
                code="frames.vector_dimension_mismatch",
                message="all vectors must have equal dimension",
            )
        for coordinate_index, entry in enumerate(vector):
            entry_location = (*vector_location, coordinate_index)
            if type(entry) is not int:
                raise OperationDomainValidationError(
                    location=entry_location,
                    code="frames.vector_entry_type",
                    message="vector entries must be exact integers",
                )
            if abs(entry) > _MAX_VECTOR_ENTRY:
                raise OperationDomainValidationError(
                    location=entry_location,
                    code="frames.vector_entry_out_of_range",
                    message="vector entries must be bounded",
                )
    assert isinstance(value, VectorFamily)
    return value


def _admit_mub(
    dimension: object, bases: object
) -> tuple[int, tuple[ComplexFrame, ...]]:
    """Admit MUB axes before indexing basis vectors."""

    if type(dimension) is not int or not 1 <= dimension <= MAX_DIM:
        raise OperationDomainValidationError(
            location=("dimension",),
            code="frames.mub_dimension",
            message="MUB dimension must be a positive bounded integer",
        )
    if type(bases) is not tuple:
        raise OperationDomainValidationError(
            location=("bases",),
            code="frames.mub_bases",
            message="MUB bases must be an exact tuple",
        )
    if not 1 <= len(bases) <= MAX_COMPLEX_BASIS_COUNT:
        raise OperationDomainValidationError(
            location=("bases",),
            code="frames.mub_basis_count",
            message=(
                "MUB requests must contain between one and "
                f"{MAX_COMPLEX_BASIS_COUNT} bases"
            ),
        )
    for basis_index, basis in enumerate(bases):
        _admit_complex_frame(basis, location=("bases", basis_index))
        assert type(basis) is ComplexFrame
        if basis.dimension != dimension or len(basis.vectors) != dimension:
            raise OperationDomainValidationError(
                location=("bases", basis_index),
                code="frames.mub_basis_shape",
                message="every basis must have exactly dimension vectors of that dimension",
            )
    assert type(dimension) is int
    assert type(bases) is tuple
    return dimension, bases


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


def _sum_shared_denominator_heights(
    values: tuple[RationalHeight, ...],
) -> RationalHeight:
    """Bound a sum, collapsing equal denominator widths to one common denominator."""

    if not values:
        return RationalHeight(1, 1)
    widths = {item.denominator_digits for item in values}
    if len(widths) != 1:
        return sum_heights(values)
    denominator_digits = next(iter(widths))
    numerator_digits = max(item.numerator_digits for item in values) + len(
        str(len(values))
    )
    return RationalHeight(numerator_digits, denominator_digits)


def _scalar_is_zero(scalar: GaussianRational) -> bool:
    return scalar.real.num == 0 and scalar.imaginary.num == 0


def _hermitian_term_heights(
    left: GaussianRational, right: GaussianRational
) -> tuple[RationalHeight, RationalHeight]:
    left_real = RationalHeight.from_canonical(left.real)
    left_imaginary = RationalHeight.from_canonical(left.imaginary)
    right_real = RationalHeight.from_canonical(right.real)
    right_imaginary = RationalHeight.from_canonical(right.imaginary)
    real = _sum_shared_denominator_heights(
        (left_real.product(right_real), left_imaginary.product(right_imaginary))
    )
    imaginary = _sum_shared_denominator_heights(
        (left_imaginary.product(right_real), left_real.product(right_imaginary))
    )
    return real, imaginary


def _sparse_inner_product_height(
    left: tuple[GaussianRational, ...], right: tuple[GaussianRational, ...]
) -> RationalHeight:
    real_terms: list[RationalHeight] = []
    imaginary_terms: list[RationalHeight] = []
    for first, second in zip(left, right, strict=True):
        if _scalar_is_zero(first) or _scalar_is_zero(second):
            continue
        real_height, imaginary_height = _hermitian_term_heights(first, second)
        real_terms.append(real_height)
        imaginary_terms.append(imaginary_height)
    real = _sum_shared_denominator_heights(tuple(real_terms))
    imaginary = _sum_shared_denominator_heights(tuple(imaginary_terms))
    return RationalHeight(
        max(real.numerator_digits, imaginary.numerator_digits),
        max(real.denominator_digits, imaginary.denominator_digits),
    )


def _frame_inner_product_height(frame: ComplexFrame) -> RationalHeight:
    pairs = [
        _sparse_inner_product_height(left, right)
        for left in frame.vectors
        for right in frame.vectors
    ]
    if not pairs:
        return RationalHeight(1, 1)
    return RationalHeight(
        max(item.numerator_digits for item in pairs),
        max(item.denominator_digits for item in pairs),
    )


def _require_complex_accumulation_height(
    frame: ComplexFrame,
    *,
    normalized_operator: bool,
    normalized_overlaps: bool,
    inner_product_output: bool,
    estimate_operator: bool = True,
) -> None:
    """Admit rational growth before constructing any exact arithmetic values."""

    source = _maximum_component_height(frame)
    product = _complex_product_height(source)
    inner_product = _frame_inner_product_height(frame)
    norm = inner_product

    if estimate_operator:
        operator_term = product
        if normalized_operator:
            operator_term = product.quotient(norm)
        operator = _complex_sum_height(operator_term, len(frame.vectors))
        trace = _complex_sum_height(operator, frame.dimension)
        scalar = trace.quotient(RationalHeight(len(str(frame.dimension)), 1))
        residual = sum_heights((operator, scalar))
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

    if inner_product_output and inner_product.exceeds(
        MAX_GAUSSIAN_RATIONAL_COMPONENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("frame", "vectors"),
            code="frames.complex_inner_product_height",
            message="Gaussian-rational inner products exceed their admitted height",
        )

    if normalized_overlaps:
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
    inner_product_output: bool = False,
    estimate_operator: bool = True,
) -> tuple[Fraction, ...]:
    if len(frame.vectors) * frame.dimension > MAX_COMPLEX_PROFILE_CELLS:
        raise OperationResourceAdmissionError(
            location=("frame", "vectors"),
            code="frames.complex_profile_cells",
            message="complex profile cells exceed the admitted output envelope",
        )
    if any(
        canonical_rational_component_digits(component) > MAX_COMPLEX_COMPONENT_DIGITS
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
        inner_product_output=inner_product_output,
        estimate_operator=estimate_operator,
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
    value = _admit_vector_family(value)
    _require_gram_work_budget(value)
    return _gram_result(value)


def verify_gram(claim: GramResult) -> bool:
    """Verify the retained vector-family Gram relation for a decoded claim."""
    try:
        _admit_vector_family(claim, expected_type=GramResult)
        if (
            claim.gram.row_count != len(claim.vectors)
            or claim.gram.column_count != len(claim.vectors)
            or any(len(vector) != claim.dimension for vector in claim.vectors)
        ):
            return False
    except (AttributeError, IndexError, TypeError, OperationDomainValidationError):
        return False
    _require_gram_work_budget(claim)
    return claim.gram.entries == integer_gram(claim.vectors)


def coherence(value: VectorFamily) -> CoherenceResult:
    """Compute exact normalized squared coherence of a finite frame."""
    value = _admit_vector_family(value)
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
    value = _admit_vector_family(value)
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

    value = _admit_vector_family(value)
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


def complex_frame_profile(frame: ComplexFrame) -> ComplexFrameProfileResult:
    """Return an exact complex frame operator and tight/equiangular profile."""

    _admit_complex_frame(frame)
    dimension = frame.dimension
    _require_complex_profile_work(frame, emitted_matrix_cells=2 * dimension * dimension)
    norms = _complex_frame_admitted(frame, normalized_overlaps=True)
    equiangular, common = _equiangular_complex_frame(frame, norms)
    tight, frame_operator, tight_residual = _tight_complex_frame(frame)
    return ComplexFrameProfileResult._from_kernel(
        frame,
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
    dimension: int, bases: tuple[ComplexFrame, ...]
) -> MutuallyUnbiasedBasesResult:
    """Decide exact mutual unbiasedness of a bounded complex basis family."""

    dimension, bases = _admit_mub(dimension, bases)
    basis_count = len(bases)
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
        _complex_frame_admitted(
            basis,
            normalized_overlaps=True,
            inner_product_output=True,
            estimate_operator=False,
        )
        for basis in bases
    )
    unbiased = True
    basis_grams: list[tuple[tuple[GaussianRational, ...], ...]] = []
    for basis in bases:
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
    for basis, norms in zip(bases, basis_norms, strict=True):
        for left_index in range(dimension):
            for right_index in range(left_index + 1, dimension):
                if _inner_product(
                    basis.vectors[left_index], basis.vectors[right_index]
                ) != (Fraction(0), Fraction(0)):
                    unbiased = False
        if any(norm <= 0 for norm in norms):
            unbiased = False
    for first in range(len(bases)):
        for second in range(first + 1, len(bases)):
            cross: list[tuple[CanonicalRational, ...]] = []
            for left_vector, left_norm in zip(
                bases[first].vectors, basis_norms[first], strict=True
            ):
                row: list[CanonicalRational] = []
                for right_vector, right_norm in zip(
                    bases[second].vectors, basis_norms[second], strict=True
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
        dimension=dimension,
        bases=bases,
        basis_grams=tuple(basis_grams),
        cross_gram_squared=tuple(cross_gram_squared),
        is_mutually_unbiased=unbiased,
    )


def sic_profile(frame: ComplexFrame) -> SicProfileResult:
    """Decide the exact SIC overlap equations for a complex vector family."""

    _admit_complex_frame(frame)
    _require_complex_profile_work(
        frame,
        emitted_matrix_cells=len(frame.vectors) ** 2
        + 2 * frame.dimension * frame.dimension,
    )
    norms = _complex_frame_admitted(
        frame, normalized_operator=True, normalized_overlaps=True
    )
    expected_count = frame.dimension * frame.dimension
    cardinality_residual = len(frame.vectors) - expected_count
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
    equiangular = True
    for left_index in range(len(squared_overlaps)):
        for right_index in range(left_index + 1, len(squared_overlaps)):
            overlap = squared_overlaps[left_index][right_index].as_fraction()
            if common is None:
                common = overlap
            elif overlap != common:
                equiangular = False
    if not equiangular:
        common = None
    expected_overlap = Fraction(1, frame.dimension + 1)
    # A one-dimensional, one-line SIC has no off-diagonal pair from which to
    # observe a common value; its defining target is still canonical.
    if common is None and cardinality_residual != 0:
        equiangular = False
    if common is None and equiangular and cardinality_residual == 0:
        common = expected_overlap
        equiangular = True
    common_residual = (
        CanonicalRational.from_fraction(common - expected_overlap)
        if common is not None
        else None
    )
    tight, frame_operator, tight_residual = _tight_complex_frame(
        frame, normalized_projectors=True, norms=norms
    )
    is_sic = (
        cardinality_residual == 0
        and equiangular
        and common_residual is not None
        and common_residual.as_fraction() == 0
        and tight
    )
    return SicProfileResult._from_kernel(
        frame,
        is_sic=is_sic,
        cardinality_residual=cardinality_residual,
        equiangular=equiangular,
        common_squared_overlap=(
            CanonicalRational.from_fraction(common) if common is not None else None
        ),
        common_squared_overlap_residual=common_residual,
        squared_overlaps=tuple(squared_overlaps),
        frame_operator=frame_operator,
        tight_residual=tight_residual,
    )
