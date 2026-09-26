"""Bounded exact theta-series prefixes for integral positive-definite forms."""

from __future__ import annotations

from itertools import permutations, product
from math import factorial, isqrt

from pydantic import ValidationError

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    MAX_THETA_PREFIX_DIMENSION,
    MAX_THETA_PREFIX_OUTPUT_DIGITS,
    MAX_THETA_PREFIX_VECTORS,
    MAX_THETA_PREFIX_WORK,
    MAX_THETA_REPRESENTATION_VECTOR_COUNT,
    MAX_THETA_SELECTED_INDEX,
    MAX_THETA_SELECTED_INDICES,
    ThetaRepresentingVectorsRequest,
    ThetaRepresentingVectorsResult,
    ThetaRepresentingVectorsRow,
    ThetaSelectedCoefficient,
    ThetaSelectedCoefficientsRequest,
    ThetaSelectedCoefficientsResult,
    ThetaSeriesPrefixRequest,
    ThetaSeriesPrefixResult,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
    QuadraticCrossTerm,
    RationalCoordinateVector,
    RationalQuadraticForm,
)


def _require_bounded_form_structure(form: RationalQuadraticForm) -> None:
    """Bound forged nested values before model serialization or validation."""
    if not isinstance(form, RationalQuadraticForm):
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.theta_invalid_form",
            message="theta coefficients require a rational quadratic form",
        )
    axis = getattr(form, "axis", None)
    diagonal = getattr(form, "diagonal_coefficients", None)
    crosses = getattr(form, "cross_terms", None)
    max_support = MAX_THETA_PREFIX_DIMENSION * (MAX_THETA_PREFIX_DIMENSION + 1) // 2
    if (
        type(getattr(form, "domain", None)) is not str
        or getattr(form, "domain", None) != "QQ"
        or type(axis) is not tuple
        or len(axis) > MAX_THETA_PREFIX_DIMENSION
        or type(diagonal) is not tuple
        or len(diagonal) != len(axis)
        or type(crosses) is not tuple
        or len(crosses) > max_support
        or any(type(label) is not str or not 1 <= len(label) <= 64 for label in axis)
    ):
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.theta_invalid_form",
            message="theta form shape exceeds its bounded canonical structure",
        )
    coefficients = list(diagonal)
    for term in crosses:
        if not isinstance(term, QuadraticCrossTerm):
            raise OperationDomainValidationError(
                location=("form", "cross_terms"),
                code="quadratic_form.theta_invalid_form",
                message="theta cross terms must be canonical bounded values",
            )
        left = getattr(term, "left", None)
        right = getattr(term, "right", None)
        if (
            type(left) is not int
            or type(right) is not int
            or not (0 <= left < right < len(axis))
        ):
            raise OperationDomainValidationError(
                location=("form", "cross_terms"),
                code="quadratic_form.theta_invalid_form",
                message="theta cross-term indices must lie in the form axis",
            )
        coefficients.append(getattr(term, "coefficient", None))
    for coefficient in coefficients:
        if not isinstance(coefficient, CanonicalRational):
            raise OperationDomainValidationError(
                location=("form",),
                code="quadratic_form.theta_invalid_form",
                message="theta coefficients must be canonical rational values",
            )
        numerator = getattr(coefficient, "num", None)
        denominator = getattr(coefficient, "den", None)
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or denominator <= 0
            or max(abs(numerator).bit_length(), denominator.bit_length())
            > 3 * MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS
        ):
            raise OperationDomainValidationError(
                location=("form",),
                code="quadratic_form.theta_invalid_form",
                message="theta coefficient components exceed the bounded integer form",
            )


def _revalidate_request(request: object, request_type: type, label: str):
    """Re-establish bounded request invariants for native callers too."""
    try:
        return request_type.model_validate(
            request.model_dump(mode="python"), strict=True
        )
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise OperationDomainValidationError(
            location=(label,),
            code="quadratic_form.theta_invalid_request",
            message="theta request must satisfy its canonical bounded schema",
        ) from exc


def _determinant(rows: tuple[tuple[int, ...], ...]) -> int:
    """Exact Leibniz determinant on the admitted dimension (at most seven)."""
    size = len(rows)
    if size == 0:
        return 1
    total = 0
    for permutation in permutations(range(size)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(size)
            for right in range(left + 1, size)
        )
        term = 1
        for row, column in enumerate(permutation):
            term *= rows[row][column]
        total += -term if inversions % 2 else term
    return total


def _minor(
    rows: tuple[tuple[int, ...], ...], removed_row: int, removed_column: int
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(value for column, value in enumerate(row) if column != removed_column)
        for index, row in enumerate(rows)
        if index != removed_row
    )


def _require_input_envelope(
    form: RationalQuadraticForm,
) -> tuple[int, int, int, int]:
    dimension = len(form.axis)
    support = dimension + len(form.cross_terms)
    if dimension > MAX_THETA_PREFIX_DIMENSION:
        raise OperationResourceAdmissionError(
            location=("form", "axis"),
            code="quadratic_form.theta_dimension_bound",
            message=f"theta prefix dimension exceeds {MAX_THETA_PREFIX_DIMENSION}",
        )
    if support > MAX_THETA_PREFIX_DIMENSION * (MAX_THETA_PREFIX_DIMENSION + 1) // 2:
        raise OperationResourceAdmissionError(
            location=("form", "cross_terms"),
            code="quadratic_form.theta_support_bound",
            message="theta prefix form support exceeds its dimension envelope",
        )
    coefficients = (
        *form.diagonal_coefficients,
        *(term.coefficient for term in form.cross_terms),
    )
    if any(value.den != 1 for value in coefficients):
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.theta_requires_integral_coefficients",
            message="theta prefixes require integral polynomial coefficients",
        )
    maximum_entry_digits = max(
        (len(str(abs(value.num * 2))) for value in form.diagonal_coefficients),
        default=1,
    )
    maximum_entry_digits = max(
        maximum_entry_digits,
        max(
            (len(str(abs(term.coefficient.num))) for term in form.cross_terms),
            default=1,
        ),
    )
    determinant_digit_bound = (
        dimension * maximum_entry_digits + len(str(max(factorial(dimension), 1))) + 2
    )
    if determinant_digit_bound > 2_000:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="quadratic_form.theta_determinant_growth",
            message="theta exact determinant intermediates exceed the digit envelope",
        )
    determinant_work = sum(size * factorial(size) for size in range(1, dimension + 1))
    cofactor_work = (
        dimension * dimension * max(dimension - 1, 0) * factorial(max(dimension - 1, 0))
    )
    if determinant_work + cofactor_work > MAX_THETA_PREFIX_WORK:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="quadratic_form.theta_determinant_work",
            message="theta exact determinant work exceeds the operation envelope",
        )
    return dimension, support, determinant_work, cofactor_work


def _positive_definite_matrix(
    form: RationalQuadraticForm, dimension: int
) -> tuple[tuple[tuple[int, ...], ...], int, tuple[int, ...]]:
    # C has integral entries under this scaling, including odd cross terms.
    matrix = [[0] * dimension for _ in range(dimension)]
    for index, coefficient in enumerate(form.diagonal_coefficients):
        matrix[index][index] = 2 * coefficient.num
    for term in form.cross_terms:
        matrix[term.left][term.right] = term.coefficient.num
        matrix[term.right][term.left] = term.coefficient.num
    integer_matrix = tuple(tuple(row) for row in matrix)

    # Sylvester's criterion establishes strict positive definiteness exactly.
    for size in range(1, dimension + 1):
        leading = tuple(tuple(row[:size]) for row in integer_matrix[:size])
        if _determinant(leading) <= 0:
            raise OperationDomainValidationError(
                location=("form",),
                code="quadratic_form.theta_requires_positive_definite",
                message="theta prefixes require a positive-definite form",
            )

    determinant = _determinant(integer_matrix)
    if dimension and determinant <= 0:
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.theta_requires_positive_definite",
            message="theta prefixes require a positive-definite form",
        )
    diagonal_cofactors = tuple(
        _determinant(_minor(integer_matrix, index, index)) for index in range(dimension)
    )
    if dimension and any(value <= 0 for value in diagonal_cofactors):
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.theta_internal_positive_definite",
            message="positive-definite cofactor invariant failed",
        )
    return integer_matrix, determinant, diagonal_cofactors


def _admit_box_and_output(
    request: (
        ThetaSeriesPrefixRequest
        | ThetaSelectedCoefficientsRequest
        | ThetaRepresentingVectorsRequest
    ),
    support: int,
    determinant_work: int,
    cofactor_work: int,
    determinant: int,
    diagonal_cofactors: tuple[int, ...],
) -> tuple[int, ...]:
    request_location = (
        ("cutoff",) if isinstance(request, ThetaSeriesPrefixRequest) else ("indices",)
    )
    radii = tuple(
        isqrt((2 * request.cutoff * cofactor) // determinant)
        for cofactor in diagonal_cofactors
    )
    vector_count = 1
    for radius in radii:
        vector_count *= 2 * radius + 1
        if vector_count > MAX_THETA_PREFIX_VECTORS:
            raise OperationResourceAdmissionError(
                location=request_location,
                code="quadratic_form.theta_vector_bound",
                message=(
                    "the proved lattice box exceeds the theta vector limit "
                    f"of {MAX_THETA_PREFIX_VECTORS}"
                ),
            )
    evaluation_work = vector_count * max(1, support)
    if determinant_work + cofactor_work + evaluation_work > MAX_THETA_PREFIX_WORK:
        raise OperationResourceAdmissionError(
            location=request_location,
            code="quadratic_form.theta_work_bound",
            message="theta enumeration exceeds the operation work envelope",
        )
    count_digits = len(str(vector_count))
    # The result retains the source form and the coefficient prefix. Bound
    # both by their aggregate decimal digits; per-entry serialization
    # structure scales with the already bounded coefficient count.
    form = request.form
    source_digits = sum(len(label) for label in form.axis) + 2 * sum(
        canonical_rational_component_digits(value)
        for value in (
            *form.diagonal_coefficients,
            *(term.coefficient for term in form.cross_terms),
        )
    )
    coefficient_count = (
        request.cutoff + 1
        if isinstance(request, ThetaSeriesPrefixRequest)
        else len(request.indices)
    )
    index_digits = (
        0
        if isinstance(request, ThetaSeriesPrefixRequest)
        else sum(len(str(index)) for index in request.indices)
    )
    if isinstance(request, ThetaRepresentingVectorsRequest):
        if vector_count > MAX_THETA_REPRESENTATION_VECTOR_COUNT:
            raise OperationResourceAdmissionError(
                location=request_location,
                code="quadratic_form.theta_representation_vector_bound",
                message="representation vectors exceed their admitted cardinality bound",
            )
    else:
        output_digits = (
            source_digits + coefficient_count * (count_digits + 1) + index_digits
        )
        if output_digits > MAX_THETA_PREFIX_OUTPUT_DIGITS:
            raise OperationResourceAdmissionError(
                location=request_location,
                code="quadratic_form.theta_output_bound",
                message="theta prefix exceeds its admitted aggregate output digit envelope",
            )
    return radii


def theta_series_prefix(
    request: ThetaSeriesPrefixRequest,
) -> ThetaSeriesPrefixResult:
    """Return every coefficient through q^N, with a proved finite search box.

    If C is twice the half-polar Gram matrix, then Q(x)=x^T C x/2. For
    positive-definite C, Cauchy-Schwarz in the C inner product gives
    x_i^2 <= (C^-1)_ii * x^T C x <= 2N*(C^-1)_ii for every vector with
    Q(x)<=N. The exact adjugate diagonal therefore yields a complete box.
    """
    request = _revalidate_request(request, ThetaSeriesPrefixRequest, "request")
    form = request.form
    dimension, support, determinant_work, cofactor_work = _require_input_envelope(form)
    _, determinant, diagonal_cofactors = _positive_definite_matrix(form, dimension)
    radii = _admit_box_and_output(
        request,
        support,
        determinant_work,
        cofactor_work,
        determinant,
        diagonal_cofactors,
    )

    table = [0] * (request.cutoff + 1)
    diagonal = tuple(value.num for value in form.diagonal_coefficients)
    crosses = tuple(
        (term.left, term.right, term.coefficient.num) for term in form.cross_terms
    )
    ranges = tuple(range(-radius, radius + 1) for radius in radii)
    for vector in product(*ranges):
        value = sum(
            coefficient * coordinate * coordinate
            for coefficient, coordinate in zip(diagonal, vector, strict=True)
        )
        value += sum(
            coefficient * vector[left] * vector[right]
            for left, right, coefficient in crosses
        )
        if 0 <= value <= request.cutoff:
            table[value] += 1
    return ThetaSeriesPrefixResult(
        form=form, cutoff=request.cutoff, coefficients=tuple(table)
    )


def theta_selected_coefficients(
    form: RationalQuadraticForm | ThetaSelectedCoefficientsRequest,
    indices: tuple[int, ...] | None = None,
) -> ThetaSelectedCoefficientsResult:
    """Return only requested r_Q(n), without constructing intervening terms."""
    if isinstance(form, ThetaSelectedCoefficientsRequest) and indices is None:
        raw_form = getattr(form, "form", None)
        raw_indices = getattr(form, "indices", None)
    else:
        raw_form = form
        raw_indices = indices
    _require_bounded_form_structure(raw_form)
    if (
        type(raw_indices) is not tuple
        or not 1 <= len(raw_indices) <= MAX_THETA_SELECTED_INDICES
        or any(
            type(index) is not int or not 0 <= index <= MAX_THETA_SELECTED_INDEX
            for index in raw_indices
        )
        or tuple(sorted(set(raw_indices))) != raw_indices
    ):
        raise OperationDomainValidationError(
            location=("indices",),
            code="quadratic_form.theta_invalid_selected_indices",
            message="selected theta indices must be bounded and strictly increasing",
        )
    try:
        form = RationalQuadraticForm.model_validate(
            raw_form.model_dump(mode="python"), strict=True
        )
    except (AttributeError, TypeError, ValueError, ValidationError) as error:
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.theta_invalid_form",
            message="selected theta coefficients received a structurally invalid form",
        ) from error
    request = ThetaSelectedCoefficientsRequest(form=form, indices=raw_indices)
    dimension, support, determinant_work, cofactor_work = _require_input_envelope(form)
    _, determinant, diagonal_cofactors = _positive_definite_matrix(form, dimension)
    radii = _admit_box_and_output(
        request,
        support,
        determinant_work,
        cofactor_work,
        determinant,
        diagonal_cofactors,
    )

    wanted = set(request.indices)
    counts = dict.fromkeys(request.indices, 0)
    diagonal = tuple(value.num for value in form.diagonal_coefficients)
    crosses = tuple(
        (term.left, term.right, term.coefficient.num) for term in form.cross_terms
    )
    ranges = tuple(range(-radius, radius + 1) for radius in radii)
    for vector in product(*ranges):
        value = sum(
            coefficient * coordinate * coordinate
            for coefficient, coordinate in zip(diagonal, vector, strict=True)
        )
        value += sum(
            coefficient * vector[left] * vector[right]
            for left, right, coefficient in crosses
        )
        if value in wanted:
            counts[value] += 1
    return ThetaSelectedCoefficientsResult(
        form=form,
        coefficients=tuple(
            ThetaSelectedCoefficient(index=index, coefficient=counts[index])
            for index in request.indices
        ),
    )


def theta_representing_vectors(
    request: ThetaRepresentingVectorsRequest,
) -> ThetaRepresentingVectorsResult:
    """Return every integer vector at each selected value, in axis order."""
    request = _revalidate_request(request, ThetaRepresentingVectorsRequest, "request")
    form = request.form
    dimension, support, determinant_work, cofactor_work = _require_input_envelope(form)
    _, determinant, diagonal_cofactors = _positive_definite_matrix(form, dimension)
    radii = _admit_box_and_output(
        request,
        support,
        determinant_work,
        cofactor_work,
        determinant,
        diagonal_cofactors,
    )

    vectors_by_value: dict[int, list[RationalCoordinateVector]] = {
        index: [] for index in request.indices
    }
    diagonal = tuple(value.num for value in form.diagonal_coefficients)
    crosses = tuple(
        (term.left, term.right, term.coefficient.num) for term in form.cross_terms
    )
    ranges = tuple(range(-radius, radius + 1) for radius in radii)
    for vector in product(*ranges):
        value = sum(
            coefficient * coordinate * coordinate
            for coefficient, coordinate in zip(diagonal, vector, strict=True)
        )
        value += sum(
            coefficient * vector[left] * vector[right]
            for left, right, coefficient in crosses
        )
        selected = vectors_by_value.get(value)
        if selected is not None:
            selected.append(
                RationalCoordinateVector(
                    axis=form.axis,
                    coordinates=tuple(
                        CanonicalRational.from_integer_ratio(coordinate, 1)
                        for coordinate in vector
                    ),
                )
            )
    return ThetaRepresentingVectorsResult(
        form=form,
        rows=tuple(
            ThetaRepresentingVectorsRow(
                index=index, vectors=tuple(vectors_by_value[index])
            )
            for index in request.indices
        ),
    )


__all__ = [
    "theta_representing_vectors",
    "theta_selected_coefficients",
    "theta_series_prefix",
]
