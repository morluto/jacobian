from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.modular_forms._tools import TOOLS
from jacobian.math.number_theory.modular_forms.basis import (
    modular_form_coordinates_q_expansion,
)
from jacobian.math.number_theory.modular_forms.coordinate_arithmetic import (
    modular_form_coordinates_add,
)
from jacobian.math.number_theory.modular_forms.coordinate_arithmetic_models import (
    ModularFormCoordinatesAddRequest,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)

_BASIS = "level-one-e4-e6-monomials-v1"


def _q(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _form(coordinates: tuple[int | Fraction, ...]) -> ModularFormCoordinates:
    return ModularFormCoordinates(
        space=ModularFormSpace(level=1, weight=12, kind="M"),
        basis_id=_BASIS,
        coordinates=tuple(_q(value) for value in coordinates),
    )


def test_coordinate_addition_matches_direct_vector_and_q_expansion_oracles() -> None:
    # These are the E4^3 and E6^2 axes of the canonical M_12 basis.
    left = _form((Fraction(3, 2), -2))
    right = _form((Fraction(-1, 2), Fraction(7, 3)))

    result = modular_form_coordinates_add(left, right)

    assert result.space == left.space
    assert result.basis_id == _BASIS
    assert tuple(value.as_fraction() for value in result.coordinates) == (
        Fraction(1),
        Fraction(1, 3),
    )
    assert ModularFormCoordinates.model_validate_json(result.model_dump_json()) == result

    # Independent Eisenstein coefficient formulas give E4^3=(1,720,179280)
    # and E6^2=(1,-1008,220752) through q^2.
    expansion = modular_form_coordinates_q_expansion(result, 3)
    assert tuple(value.as_fraction() for value in expansion.q_expansion.coefficients) == (
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
        basis_id="gamma0-two-weight-2-4-monomials-v1",
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

    value = result.coordinates[0].as_fraction()
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
