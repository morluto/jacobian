"""Bounded exact arithmetic for modular-form coordinate values."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.modular_forms.basis import _admit_coordinates
from jacobian.math.number_theory.modular_forms.values import (
    MAX_LEVEL_ONE_BASIS_COEFFICIENT_DIGITS,
    MAX_LEVEL_ONE_BASIS_COORDINATES,
    ModularFormCoordinates,
)

MAX_COORDINATE_ADDITION_DIGITS = 2 * MAX_LEVEL_ONE_BASIS_COEFFICIENT_DIGITS + 1
MAX_COORDINATE_ADDITION_CELLS = MAX_LEVEL_ONE_BASIS_COORDINATES
MAX_COORDINATE_ADDITION_WORK = 3 * MAX_LEVEL_ONE_BASIS_COORDINATES
MAX_COORDINATE_ADDITION_OUTPUT_BYTES = CanonicalLimits().max_output_bytes
MAX_COORDINATE_SCALAR_DIGITS = 2 * MAX_LEVEL_ONE_BASIS_COEFFICIENT_DIGITS
MAX_COORDINATE_SCALAR_CELLS = MAX_LEVEL_ONE_BASIS_COORDINATES
MAX_COORDINATE_SCALAR_WORK = 2 * MAX_LEVEL_ONE_BASIS_COORDINATES
MAX_COORDINATE_SCALAR_OUTPUT_BYTES = CanonicalLimits().max_output_bytes


def _digits(value: int) -> int:
    return len(str(abs(value)))


def _integer_digit_upper_bound(value: int) -> int:
    """Bound decimal digits without converting an untrusted large integer to text."""
    bits = abs(value).bit_length()
    return (bits * 30_103 + 99_999) // 100_000 if bits else 1


def _require_same_parent(
    left: ModularFormCoordinates, right: ModularFormCoordinates
) -> None:
    if left.space != right.space:
        raise OperationDomainValidationError(
            location=("right", "space"),
            code="modular_form.coordinate_add_space_mismatch",
            message="both coordinate forms must have the identical exact modular space",
        )
    if left.basis_id != right.basis_id:
        raise OperationDomainValidationError(
            location=("right", "basis_id"),
            code="modular_form.coordinate_add_basis_mismatch",
            message="both coordinate forms must use the identical canonical basis",
        )
    if left.space.coefficient_domain != "QQ":
        raise OperationDomainValidationError(
            location=("left", "space", "coefficient_domain"),
            code="modular_form.coordinate_add_coefficient_domain",
            message="coordinate addition currently supports rational QQ coefficients",
        )


def _projected_digit_bound(left: Fraction, right: Fraction) -> int:
    numerator_bound = (
        max(
            _digits(left.numerator) + _digits(right.denominator),
            _digits(right.numerator) + _digits(left.denominator),
        )
        + 1
    )
    denominator_bound = _digits(left.denominator) + _digits(right.denominator)
    return max(numerator_bound, denominator_bound)


def modular_form_coordinates_add(
    left: ModularFormCoordinates,
    right: ModularFormCoordinates,
) -> ModularFormCoordinates:
    """Add two rational coordinate vectors in their shared canonical basis."""
    if not isinstance(left, ModularFormCoordinates) or not isinstance(
        right, ModularFormCoordinates
    ):
        raise OperationDomainValidationError(
            location=(),
            code="modular_form.coordinate_add_input_type",
            message="both operands must be exact modular-form coordinate values",
        )
    _require_same_parent(left, right)

    plan, left_values = _admit_coordinates(
        left,
        1,
        materialize_pari=False,
        check_expansion_growth=False,
    )
    _, right_values = _admit_coordinates(
        right,
        1,
        admitted_plan=plan,
        materialize_pari=False,
        check_expansion_growth=False,
    )

    dimension = plan.dimension
    output_cells = dimension
    work = 3 * dimension
    if output_cells > MAX_COORDINATE_ADDITION_CELLS:
        raise OperationResourceAdmissionError(
            location=("left", "coordinates"),
            code="modular_form.coordinate_add_cell_bound",
            message="coordinate addition output exceeds the admitted cell bound",
        )
    if work > MAX_COORDINATE_ADDITION_WORK:
        raise OperationResourceAdmissionError(
            location=("left", "coordinates"),
            code="modular_form.coordinate_add_work_bound",
            message="coordinate addition exceeds the admitted exact work bound",
        )
    projected_digits = max(
        (
            _projected_digit_bound(a, b)
            for a, b in zip(left_values, right_values, strict=True)
        ),
        default=1,
    )
    if projected_digits > MAX_COORDINATE_ADDITION_DIGITS:
        raise OperationResourceAdmissionError(
            location=("left", "coordinates"),
            code="modular_form.coordinate_add_digit_bound",
            message="predicted rational coordinate growth exceeds the addition envelope",
        )

    parent_bytes = len(encode_strict_json(left.space.model_dump(mode="json")))
    basis_bytes = len(encode_strict_json(left.basis_id))
    output_bytes = (
        256 + parent_bytes + basis_bytes + output_cells * (2 * projected_digits + 80)
    )
    if output_bytes > MAX_COORDINATE_ADDITION_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("left", "coordinates"),
            code="modular_form.coordinate_add_output_bound",
            message="predicted coordinate result exceeds the canonical output limit",
        )

    result = tuple(a + b for a, b in zip(left_values, right_values, strict=True))
    return ModularFormCoordinates(
        space=left.space,
        basis_id=left.basis_id,
        coordinates=tuple(CanonicalRational.from_fraction(value) for value in result),
    )


def modular_form_coordinates_scalar_multiply(
    form: ModularFormCoordinates, scalar: CanonicalRational
) -> ModularFormCoordinates:
    """Scale a rational coordinate vector, preserving its exact parent and axes."""
    if not isinstance(form, ModularFormCoordinates) or not isinstance(
        scalar, CanonicalRational
    ):
        raise OperationDomainValidationError(
            location=(),
            code="modular_form.coordinate_scalar_input_type",
            message="scaling requires exact modular-form coordinates and a rational scalar",
        )
    if form.space.coefficient_domain != "QQ":
        raise OperationDomainValidationError(
            location=("form", "space", "coefficient_domain"),
            code="modular_form.coordinate_scalar_coefficient_domain",
            message="coordinate scaling currently supports rational QQ coefficients",
        )

    plan, values = _admit_coordinates(
        form,
        1,
        materialize_pari=False,
        check_expansion_growth=False,
    )
    numerator, denominator = scalar.as_integer_ratio()
    scalar_digits = max(
        _integer_digit_upper_bound(numerator),
        _integer_digit_upper_bound(denominator),
    )
    if scalar_digits > MAX_LEVEL_ONE_BASIS_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("scalar",),
            code="modular_form.coordinate_scalar_digit_bound",
            message="scalar exceeds the admitted rational digit bound",
        )

    scalar_value = Fraction(numerator, denominator)

    dimension = plan.dimension
    if (
        dimension > MAX_COORDINATE_SCALAR_CELLS
        or 2 * dimension > MAX_COORDINATE_SCALAR_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("form", "coordinates"),
            code="modular_form.coordinate_scalar_work_bound",
            message="coordinate scaling exceeds the admitted exact work bound",
        )
    projected_digits = max(
        (
            max(
                _digits(value.numerator) + _digits(scalar_value.numerator),
                _digits(value.denominator) + _digits(scalar_value.denominator),
            )
            for value in values
        ),
        default=1,
    )
    if projected_digits > MAX_COORDINATE_SCALAR_DIGITS:
        raise OperationResourceAdmissionError(
            location=("form", "coordinates"),
            code="modular_form.coordinate_scalar_digit_bound",
            message="predicted rational coordinate growth exceeds the scaling envelope",
        )

    parent_bytes = len(encode_strict_json(form.space.model_dump(mode="json")))
    basis_bytes = len(encode_strict_json(form.basis_id))
    output_bytes = (
        256 + parent_bytes + basis_bytes + dimension * (2 * projected_digits + 80)
    )
    if output_bytes > MAX_COORDINATE_SCALAR_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("form", "coordinates"),
            code="modular_form.coordinate_scalar_output_bound",
            message="predicted coordinate result exceeds the canonical output limit",
        )

    result = tuple(value * scalar_value for value in values)
    return ModularFormCoordinates(
        space=form.space,
        basis_id=form.basis_id,
        coordinates=tuple(CanonicalRational.from_fraction(value) for value in result),
    )


__all__ = [
    "MAX_COORDINATE_ADDITION_CELLS",
    "MAX_COORDINATE_ADDITION_DIGITS",
    "MAX_COORDINATE_ADDITION_OUTPUT_BYTES",
    "MAX_COORDINATE_ADDITION_WORK",
    "MAX_COORDINATE_SCALAR_CELLS",
    "MAX_COORDINATE_SCALAR_DIGITS",
    "MAX_COORDINATE_SCALAR_OUTPUT_BYTES",
    "MAX_COORDINATE_SCALAR_WORK",
    "modular_form_coordinates_add",
    "modular_form_coordinates_scalar_multiply",
]
