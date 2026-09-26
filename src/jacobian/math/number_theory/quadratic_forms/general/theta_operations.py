"""Bounded exact theta-series prefixes for integral positive-definite forms."""

from __future__ import annotations

from itertools import permutations, product
from math import factorial, isqrt

from jacobian._exact import canonical_rational_component_digits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    MAX_THETA_PREFIX_CUTOFF,
    MAX_THETA_PREFIX_DIMENSION,
    MAX_THETA_PREFIX_OUTPUT_DIGITS,
    MAX_THETA_PREFIX_VECTORS,
    MAX_THETA_PREFIX_WORK,
    MAX_THETA_SELECTED_INDEX,
    MAX_THETA_SELECTED_INDICES,
    ThetaSelectedCoefficient,
    ThetaSelectedCoefficientsResult,
    ThetaSeriesPrefixRequest,
    ThetaSeriesPrefixResult,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    RationalQuadraticForm,
)


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
    *,
    form: RationalQuadraticForm,
    cutoff: int,
    request_location: tuple[str, ...],
    coefficient_count: int,
    index_digits: int,
    support: int,
    determinant_work: int,
    cofactor_work: int,
    determinant: int,
    diagonal_cofactors: tuple[int, ...],
) -> tuple[int, ...]:
    radii = tuple(
        isqrt((2 * cutoff * cofactor) // determinant) for cofactor in diagonal_cofactors
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
    source_digits = sum(len(label) for label in form.axis) + 2 * sum(
        canonical_rational_component_digits(value)
        for value in (
            *form.diagonal_coefficients,
            *(term.coefficient for term in form.cross_terms),
        )
    )
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
    form = request.form
    if (
        not isinstance(request.cutoff, int)
        or isinstance(request.cutoff, bool)
        or not 0 <= request.cutoff <= MAX_THETA_PREFIX_CUTOFF
    ):
        raise OperationDomainValidationError(
            location=("cutoff",),
            code="quadratic_form.theta.cutoff_bound",
            message=f"cutoff must be an integer from 0 through {MAX_THETA_PREFIX_CUTOFF}",
        )
    dimension, support, determinant_work, cofactor_work = _require_input_envelope(form)
    _, determinant, diagonal_cofactors = _positive_definite_matrix(form, dimension)
    radii = _admit_box_and_output(
        form=form,
        cutoff=request.cutoff,
        request_location=("cutoff",),
        coefficient_count=request.cutoff + 1,
        index_digits=0,
        support=support,
        determinant_work=determinant_work,
        cofactor_work=cofactor_work,
        determinant=determinant,
        diagonal_cofactors=diagonal_cofactors,
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
    form: RationalQuadraticForm,
    indices: tuple[int, ...],
) -> ThetaSelectedCoefficientsResult:
    """Return only requested r_Q(n), without constructing intervening terms."""
    if not isinstance(form, RationalQuadraticForm):
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.theta_invalid_form",
            message="selected theta coefficients require a rational quadratic form",
        )
    if (
        type(indices) is not tuple
        or not 1 <= len(indices) <= MAX_THETA_SELECTED_INDICES
        or any(
            type(index) is not int or not 0 <= index <= MAX_THETA_SELECTED_INDEX
            for index in indices
        )
        or tuple(sorted(set(indices))) != indices
    ):
        raise OperationDomainValidationError(
            location=("indices",),
            code="quadratic_form.theta_invalid_selected_indices",
            message="selected theta indices must be bounded and strictly increasing",
        )
    if not (
        isinstance(form.axis, tuple)
        and isinstance(form.diagonal_coefficients, tuple)
        and isinstance(form.cross_terms, tuple)
        and len(form.axis) <= MAX_THETA_PREFIX_DIMENSION
        and len(form.diagonal_coefficients) == len(form.axis)
        and len(form.cross_terms)
        <= MAX_THETA_PREFIX_DIMENSION * (MAX_THETA_PREFIX_DIMENSION + 1) // 2
    ):
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.theta_invalid_form",
            message="selected theta coefficients require a bounded canonical form",
        )
    try:
        form = RationalQuadraticForm.model_validate(form.model_dump(), strict=True)
    except Exception as error:
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.theta_invalid_form",
            message="selected theta coefficients received a structurally invalid form",
        ) from error
    dimension, support, determinant_work, cofactor_work = _require_input_envelope(form)
    _, determinant, diagonal_cofactors = _positive_definite_matrix(form, dimension)
    radii = _admit_box_and_output(
        form=form,
        cutoff=indices[-1],
        request_location=("indices",),
        coefficient_count=len(indices),
        index_digits=sum(len(str(index)) for index in indices),
        support=support,
        determinant_work=determinant_work,
        cofactor_work=cofactor_work,
        determinant=determinant,
        diagonal_cofactors=diagonal_cofactors,
    )

    wanted = set(indices)
    counts = dict.fromkeys(indices, 0)
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
            for index in indices
        ),
    )


__all__ = ["theta_selected_coefficients", "theta_series_prefix"]
