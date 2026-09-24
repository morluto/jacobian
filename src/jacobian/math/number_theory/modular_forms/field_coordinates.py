"""Explicit scalar extension for bounded rational modular-form coordinates."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.modular_forms import cyclotomic
from jacobian.math.number_theory.modular_forms.basis import (
    MAX_COORDINATE_RESULT_DIGITS,
    MAX_LEVEL_ONE_BASIS_OUTPUT_BYTES,
    MAX_LEVEL_ONE_BASIS_WORK,
    PARI_STURM_RREF_BASIS_ID,
    _admit_basis,
    _admit_coordinates,
    _basis_coefficients,
    _materialize_pari_basis,
)
from jacobian.math.number_theory.modular_forms.transforms import sturm_bound
from jacobian.math.number_theory.modular_forms.values import (
    MAX_LEVEL_ONE_BASIS_PRECISION,
    ModularFormCoordinates,
    ModularFormFieldQExpansion,
    ModularFormSpace,
)

_FIELD = RationalCyclotomicField(order=6)


def _unsupported(message: str) -> None:
    raise OperationDomainValidationError(
        location=("space",),
        code="modular_form.field_coordinates_unsupported_space",
        message=message,
    )


def _rational_space(space: ModularFormSpace) -> ModularFormSpace:
    if space.character != "TRIVIAL":
        _unsupported("field-valued coordinates currently require trivial character")
    return ModularFormSpace(
        level=space.level,
        weight=space.weight,
        kind=space.kind,
        character="TRIVIAL",
        coefficient_domain="QQ",
    )


def _embed_rational(
    field: RationalCyclotomicField, value: int | Fraction
) -> RationalCyclotomicElement:
    numerator, denominator = (
        (value.numerator, value.denominator)
        if isinstance(value, Fraction)
        else (value, 1)
    )
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=(
            CanonicalRational(num=numerator, den=denominator),
            *(CanonicalRational(num=0, den=1) for _ in range(1, field.degree)),
        ),
    )


def modular_form_coordinates_extend_field(
    form: ModularFormCoordinates, coefficient_field: RationalCyclotomicField
) -> ModularFormCoordinates:
    """Embed exact rational coordinates into the explicit field ``Q(zeta_6)``."""
    if not isinstance(form, ModularFormCoordinates):
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.coordinates_type",
            message="field extension requires exact rational modular-form coordinates",
        )
    if (
        type(coefficient_field) is not RationalCyclotomicField
        or coefficient_field != _FIELD
    ):
        raise OperationDomainValidationError(
            location=("coefficient_field",),
            code="modular_form.field_coordinates_field",
            message="this scalar-extension slice currently supports exactly Q(zeta_6)",
        )
    rational_space = _rational_space(form.space)
    if form.space != rational_space:
        _unsupported("source coordinates must have rational coefficient parent QQ")
    precision = sturm_bound(rational_space).bound + 1
    plan, coordinates = _admit_coordinates(
        form,
        precision,
        materialize_pari=False,
        check_expansion_growth=False,
    )
    request_checkpoint("before exact modular-form scalar extension")
    zero = CanonicalRational(num=0, den=1)
    lifted = tuple(
        RationalCyclotomicElement(
            field=coefficient_field,
            coefficients_ascending=(
                CanonicalRational(num=value.numerator, den=value.denominator),
                zero,
            ),
        )
        for value in coordinates
    )
    return ModularFormCoordinates(
        space=ModularFormSpace(
            level=form.space.level,
            weight=form.space.weight,
            kind=form.space.kind,
            character="TRIVIAL",
            coefficient_domain=coefficient_field,
        ),
        basis_id=plan.basis_id,
        coordinates=lifted,
    )


def modular_form_field_coordinates_q_expansion(
    form: ModularFormCoordinates, precision: int
) -> ModularFormFieldQExpansion:
    """Evaluate field coordinates in the rational basis with exact admission."""
    if not isinstance(form, ModularFormCoordinates):
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.field_coordinates_type",
            message="form must be an exact field-valued modular-form coordinate value",
        )
    if (
        type(precision) is not int
        or not 1 <= precision <= MAX_LEVEL_ONE_BASIS_PRECISION
    ):
        raise OperationResourceAdmissionError(
            location=("precision",),
            code="modular_form.field_coordinates_precision",
            message=(
                "field-valued q-expansion precision must lie in [1, "
                f"{MAX_LEVEL_ONE_BASIS_PRECISION}]"
            ),
        )
    rational_space = _rational_space(form.space)
    plan = _admit_basis(rational_space, precision, materialize_pari=False)
    if (
        form.basis_id != plan.basis_id
        or type(form.coordinates) is not tuple
        or len(form.coordinates) != plan.dimension
    ):
        raise OperationDomainValidationError(
            location=("form",),
            code="modular_form.field_coordinates_basis_parent",
            message="field coordinate basis and shape must match the exact rational space basis",
        )
    coordinate_digits = max(
        (cyclotomic._validate_element(value)[2] for value in form.coordinates),
        default=1,
    )
    result_digits = (
        max(1, plan.dimension) * (coordinate_digits + plan.coefficient_digits)
        + len(str(max(1, plan.dimension)))
        + 2
    )
    work = precision * max(1, plan.dimension) * form.space.coefficient_domain.degree**2
    output_bytes = (
        precision * form.space.coefficient_domain.degree * (2 * result_digits + 32)
    )
    if (
        result_digits
        > min(MAX_COORDINATE_RESULT_DIGITS, MAX_CYCLIC_FIELD_ELEMENT_DIGITS)
        or work > MAX_LEVEL_ONE_BASIS_WORK
        or output_bytes > MAX_LEVEL_ONE_BASIS_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("form",),
            code="modular_form.field_coordinates_output_bound",
            message="field-valued q-expansion exceeds its exact height, work, or output envelope",
        )
    if plan.basis_id == PARI_STURM_RREF_BASIS_ID:
        plan = _materialize_pari_basis(plan)
    basis = _basis_coefficients(plan)
    field = form.space.coefficient_domain
    zero = _embed_rational(field, 0)
    coefficients = []
    for index in range(precision):
        request_checkpoint("during field-valued modular-form q-expansion")
        value = zero
        for coordinate, vector in zip(form.coordinates, basis, strict=True):
            rational = vector[index]
            embedded = _embed_rational(field, rational)
            value = cyclotomic.add(value, cyclotomic.multiply(coordinate, embedded))
        coefficients.append(value)
    return ModularFormFieldQExpansion(
        space=form.space,
        coefficients=tuple(coefficients),
    )


def modular_form_field_coordinates_equal(
    left: ModularFormCoordinates, right: ModularFormCoordinates
) -> bool:
    """Compare exact field coordinates in one admitted trivial-character basis."""
    if not isinstance(left, ModularFormCoordinates) or not isinstance(
        right, ModularFormCoordinates
    ):
        raise OperationDomainValidationError(
            location=(),
            code="modular_form.coordinates_type",
            message="both operands must be exact modular-form coordinate values",
        )
    if left.space != right.space or left.basis_id != right.basis_id:
        raise OperationDomainValidationError(
            location=("right", "space"),
            code="modular_form.field_equality_parent",
            message="field coordinate equality requires the identical space and basis",
        )
    rational_space = _rational_space(left.space)
    plan = _admit_basis(rational_space, 1, materialize_pari=False)
    if (
        left.basis_id != plan.basis_id
        or type(left.coordinates) is not tuple
        or type(right.coordinates) is not tuple
        or len(left.coordinates) != plan.dimension
        or len(right.coordinates) != plan.dimension
    ):
        raise OperationDomainValidationError(
            location=("left", "basis_id"),
            code="modular_form.field_equality_basis",
            message="field coordinate basis and shape must match the admitted space basis",
        )
    field = left.space.coefficient_domain
    values = (*left.coordinates, *right.coordinates)
    if any(
        not isinstance(value, RationalCyclotomicElement) or value.field != field
        for value in values
    ):
        raise OperationDomainValidationError(
            location=("left", "coordinates"),
            code="modular_form.field_equality_scalar",
            message="every field coordinate must belong to the exact space coefficient field",
        )
    for value in values:
        cyclotomic._validate_element(value)
    return left.coordinates == right.coordinates


__all__ = [
    "modular_form_coordinates_extend_field",
    "modular_form_field_coordinates_equal",
    "modular_form_field_coordinates_q_expansion",
]
