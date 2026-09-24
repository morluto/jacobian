"""Complete bounded value profiles on symmetric integer boxes."""

from __future__ import annotations

from collections import Counter
from itertools import product

from jacobian._exact import canonical_rational_component_digits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    MAX_QUADRATIC_BOX_OUTPUT_DIGITS,
    MAX_QUADRATIC_BOX_PROFILE_ROWS,
    MAX_QUADRATIC_BOX_RADIUS,
    MAX_QUADRATIC_BOX_VECTORS,
    FiniteBoxProfileRequest,
    FiniteBoxProfileResult,
    FiniteBoxProfileRow,
)


def _admit(request: FiniteBoxProfileRequest) -> int:
    form = request.form
    if request.radius > MAX_QUADRATIC_BOX_RADIUS:
        raise OperationResourceAdmissionError(
            location=("radius",),
            code="quadratic_form.finite_box_radius_bound",
            message="finite-box radius exceeds the owner envelope",
        )
    if any(value.den != 1 for value in form.diagonal_coefficients) or any(
        term.coefficient.den != 1 for term in form.cross_terms
    ):
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.finite_box_requires_integral",
            message="finite-box value profiles require integral polynomial coefficients",
        )
    dimension = len(form.axis)
    support = dimension + len(form.cross_terms)
    vector_count = 1
    side_length = 2 * request.radius + 1
    for _ in range(dimension):
        if vector_count > MAX_QUADRATIC_BOX_VECTORS // side_length:
            vector_count = MAX_QUADRATIC_BOX_VECTORS + 1
            break
        vector_count *= side_length
    if vector_count > MAX_QUADRATIC_BOX_VECTORS:
        raise OperationResourceAdmissionError(
            location=("radius", "form", "axis"),
            code="quadratic_form.finite_box_vector_bound",
            message=(
                "complete integer-box enumeration exceeds the admitted vector "
                f"limit of {MAX_QUADRATIC_BOX_VECTORS}"
            ),
        )
    work = vector_count * max(support, 1)
    if work > 2_000_000:
        raise OperationResourceAdmissionError(
            location=("form", "axis"),
            code="quadratic_form.finite_box_work_bound",
            message="finite-box evaluation exceeds the admitted arithmetic work",
        )

    # |Q(x)| <= B^2 * sum |coefficients|. This integer upper bound is
    # computed from already bounded coefficients before vector evaluation.
    coefficient_magnitude = sum(
        abs(value.num) for value in form.diagonal_coefficients
    ) + sum(abs(term.coefficient.num) for term in form.cross_terms)
    value_digits = len(str(max(1, coefficient_magnitude * request.radius**2)))
    count_digits = len(str(vector_count))
    # Admit the worst case of one output row per enumerated vector by
    # aggregate decimal digits: the retained source coefficients and axis
    # labels plus two components per profile entry. Per-entry serialization
    # structure scales with the vector and profile-row cardinality bounds.
    source_digits = sum(len(label) for label in form.axis) + 2 * sum(
        canonical_rational_component_digits(value)
        for value in (
            *form.diagonal_coefficients,
            *(term.coefficient for term in form.cross_terms),
        )
    )
    profile_digits = vector_count * (value_digits + count_digits)
    if (
        vector_count > MAX_QUADRATIC_BOX_PROFILE_ROWS
        or source_digits + profile_digits > MAX_QUADRATIC_BOX_OUTPUT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("radius", "form", "axis"),
            code="quadratic_form.finite_box_output_bound",
            message="finite-box complete profile exceeds its admitted output digit envelope",
        )
    return vector_count


def finite_box_value_profile(
    request: FiniteBoxProfileRequest,
) -> FiniteBoxProfileResult:
    """Count every integral vector by its exact form value in ``[-B,B]^n``."""

    vector_count = _admit(request)
    form = request.form
    radius = request.radius
    diagonal = tuple(value.num for value in form.diagonal_coefficients)
    cross = tuple(
        (term.left, term.right, term.coefficient.num) for term in form.cross_terms
    )
    counts: Counter[int] = Counter()
    for vector in product(range(-radius, radius + 1), repeat=len(form.axis)):
        value = sum(
            coefficient * coordinate * coordinate
            for coefficient, coordinate in zip(diagonal, vector, strict=True)
        ) + sum(
            coefficient * vector[left] * vector[right]
            for left, right, coefficient in cross
        )
        counts[value] += 1
    rows = tuple(
        FiniteBoxProfileRow(value=value, representation_count=count)
        for value, count in sorted(counts.items())
    )
    return FiniteBoxProfileResult._from_kernel(
        form=form,
        radius=radius,
        coordinate_bounds=tuple((-radius, radius) for _ in form.axis),
        vector_count=vector_count,
        rows=rows,
        minimum_value=rows[0].value,
        maximum_value=rows[-1].value,
    )


__all__ = ["finite_box_value_profile"]
