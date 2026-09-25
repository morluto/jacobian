from fractions import Fraction
from typing import Literal

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.modular_forms._tools import TOOLS
from jacobian.math.number_theory.modular_forms.basis import (
    modular_form_coordinates_q_expansion,
)
from jacobian.math.number_theory.modular_forms.coordinate_arithmetic import (
    modular_form_coordinates_add,
    modular_form_coordinates_scalar_multiply,
)
from jacobian.math.number_theory.modular_forms.coordinate_arithmetic_models import (
    ModularFormCoordinatesAddRequest,
    ModularFormCoordinatesScalarMultiplyRequest,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)

_BASIS: Literal["level-one-e4-e6-monomials-v1"] = "level-one-e4-e6-monomials-v1"
_OTHER_BASIS: Literal["gamma0-two-weight-2-4-monomials-v1"] = (
    "gamma0-two-weight-2-4-monomials-v1"
)


def _q(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _form(coordinates: tuple[int | Fraction, ...]) -> ModularFormCoordinates:
    return ModularFormCoordinates(
        space=ModularFormSpace(level=1, weight=12, kind="M"),
        basis_id=_BASIS,
        coordinates=tuple(_q(value) for value in coordinates),
    )


def _fractions(form: ModularFormCoordinates) -> tuple[Fraction, ...]:
    assert all(isinstance(value, CanonicalRational) for value in form.coordinates)
    return tuple(
        value.as_fraction()
        for value in form.coordinates
        if isinstance(value, CanonicalRational)
    )


def test_coordinate_addition_matches_direct_vector_and_q_expansion_oracles() -> None:
    # These are the E4^3 and E6^2 axes of the canonical M_12 basis.
    left = _form((Fraction(3, 2), -2))
    right = _form((Fraction(-1, 2), Fraction(7, 3)))

    result = modular_form_coordinates_add(left, right)

    assert result.space == left.space
    assert result.basis_id == _BASIS
    assert _fractions(result) == (
        Fraction(1),
        Fraction(1, 3),
    )
    assert (
        ModularFormCoordinates.model_validate_json(result.model_dump_json()) == result
    )

    # Independent Eisenstein coefficient formulas give E4^3=(1,720,179280)
    # and E6^2=(1,-1008,220752) through q^2.
    expansion = modular_form_coordinates_q_expansion(result, 3)
    assert tuple(
        value.as_fraction() for value in expansion.q_expansion.coefficients
    ) == (
        Fraction(4, 3),
        Fraction(384),
        Fraction(252_864),
    )


def test_coordinate_addition_rejects_different_complete_parents() -> None:
    left = _form((1, 0))
    different_space = ModularFormCoordinates(
        space=ModularFormSpace(level=2, weight=12, kind="M"),
        basis_id=_BASIS,
        coordinates=(_q(1), _q(0)),
    )
    with pytest.raises(OperationDomainValidationError) as mismatch:
        modular_form_coordinates_add(left, different_space)
    assert mismatch.value.errors()[0]["type"] == (
        "modular_form.coordinate_add_space_mismatch"
    )


def test_coordinate_addition_rejects_different_basis_axes() -> None:
    left = _form((1, 0))
    other_basis = ModularFormCoordinates(
        space=left.space,
        basis_id=_OTHER_BASIS,
        coordinates=(_q(1), _q(0)),
    )
    with pytest.raises(OperationDomainValidationError) as mismatch:
        modular_form_coordinates_add(left, other_basis)
    assert mismatch.value.errors()[0]["type"] == (
        "modular_form.coordinate_add_basis_mismatch"
    )


def test_coordinate_addition_handles_the_zero_dimensional_space() -> None:
    zero_cusp = ModularFormCoordinates(
        space=ModularFormSpace(level=1, weight=0, kind="S"),
        basis_id=_BASIS,
        coordinates=(),
    )
    result = modular_form_coordinates_add(zero_cusp, zero_cusp)
    assert result == zero_cusp


def test_coordinate_addition_admits_projected_denominator_growth() -> None:
    first_denominator = 10**127 + 1
    second_denominator = 10**127 + 2
    left = _form((Fraction(1, first_denominator), 0))
    right = _form((Fraction(1, second_denominator), 0))

    result = modular_form_coordinates_add(left, right)

    value = _fractions(result)[0]
    assert value == Fraction(1, first_denominator) + Fraction(1, second_denominator)
    assert len(str(value.denominator)) == 255


def test_coordinate_addition_is_published_with_executable_example() -> None:
    operations = [
        operation
        for operation in TOOLS
        if operation.operation_id == "modular_form.coordinates.add.compute"
    ]
    assert len(operations) == 1
    assert operations[0].examples
    assert ModularFormCoordinatesAddRequest.model_validate_json(
        encode_strict_json(operations[0].examples[0].input), strict=True
    )


def test_coordinate_scalar_action_matches_vector_and_q_expansion_oracles() -> None:
    form = _form((1, Fraction(1, 3)))
    scalar = _q(Fraction(2, 3))

    result = modular_form_coordinates_scalar_multiply(form, scalar)

    assert result.space == form.space
    assert result.basis_id == form.basis_id
    assert _fractions(result) == (
        Fraction(2, 3),
        Fraction(2, 9),
    )
    assert (
        ModularFormCoordinates.model_validate_json(result.model_dump_json()) == result
    )
    expansion = modular_form_coordinates_q_expansion(result, 3)
    assert tuple(
        value.as_fraction() for value in expansion.q_expansion.coefficients
    ) == (
        Fraction(8, 9),
        Fraction(256),
        Fraction(168_576),
    )


def test_scalar_digit_count_and_reusable_coordinate_envelope() -> None:
    result = modular_form_coordinates_scalar_multiply(_form((1, 0)), _q(9 * 10**127))
    assert _fractions(result) == (Fraction(9 * 10**127), Fraction(0))
    with pytest.raises(OperationResourceAdmissionError):
        modular_form_coordinates_scalar_multiply(_form((10**127, 0)), _q(10**127))


def test_forged_noncanonical_scalar_is_rejected_at_native_boundary() -> None:
    form = _form((1,))
    forged = CanonicalRational.model_construct(num=2, den=4)
    with pytest.raises(OperationDomainValidationError) as rejected:
        modular_form_coordinates_scalar_multiply(form, forged)
    assert rejected.value.errors()[0]["type"] == (
        "modular_form.coordinate_scalar_invalid_rational"
    )
    zero_denominator = CanonicalRational.model_construct(num=1, den=0)
    with pytest.raises(OperationDomainValidationError):
        modular_form_coordinates_scalar_multiply(form, zero_denominator)


def test_coordinate_scalar_action_handles_zero_scalar_and_empty_space() -> None:
    form = _form((Fraction(7, 5), -2))
    assert modular_form_coordinates_scalar_multiply(form, _q(0)) == _form((0, 0))

    zero_cusp = ModularFormCoordinates(
        space=ModularFormSpace(level=1, weight=0, kind="S"),
        basis_id=_BASIS,
        coordinates=(),
    )
    assert modular_form_coordinates_scalar_multiply(zero_cusp, _q(13)) == zero_cusp


def test_coordinate_scalar_action_preflights_scalar_digits() -> None:
    form = _form((1, 0))
    too_large = _q(10**128)
    with pytest.raises(OperationResourceAdmissionError) as rejected:
        modular_form_coordinates_scalar_multiply(form, too_large)
    assert rejected.value.errors()[0]["type"] == (
        "modular_form.coordinate_scalar_digit_bound"
    )


def test_coordinate_scalar_action_is_published_with_executable_example() -> None:
    operations = [
        operation
        for operation in TOOLS
        if operation.operation_id == "modular_form.coordinates.scalar_multiply.compute"
    ]
    assert len(operations) == 1
    assert operations[0].examples
    request = ModularFormCoordinatesScalarMultiplyRequest.model_validate_json(
        encode_strict_json(operations[0].examples[0].input), strict=True
    )
    result = operations[0].run(request)
    assert tuple(value.as_fraction() for value in result.coordinates) == (2, 0)
