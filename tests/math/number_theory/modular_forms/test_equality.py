from typing import Literal, cast

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.modular_forms._tools import TOOLS
from jacobian.math.number_theory.modular_forms.basis import (
    modular_form_coordinates_equal,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)

BasisId = Literal[
    "level-one-e4-e6-monomials-v1",
    "gamma0-two-weight-2-4-monomials-v1",
    "gamma0-three-weight-2-4-6-hypersurface-v1",
    "gamma0-four-weight-2-generators-v1",
    "gamma0-four-chi4-weight-one-v1",
    "gamma0-four-chi4-weight-three-v1",
]


def _form(level: int, weight: int, coords: tuple[int, ...]) -> ModularFormCoordinates:
    space = ModularFormSpace(level=level, weight=weight, kind="M")
    basis_ids = {
        (1, 4): "level-one-e4-e6-monomials-v1",
        (1, 12): "level-one-e4-e6-monomials-v1",
        (2, 4): "gamma0-two-weight-2-4-monomials-v1",
    }
    return ModularFormCoordinates(
        space=space,
        basis_id=cast(BasisId, basis_ids[level, weight]),
        coordinates=tuple(CanonicalRational(num=value, den=1) for value in coords),
    )


def test_global_equality_compares_admitted_canonical_coordinates() -> None:
    left = _form(2, 4, (2, 3))
    same = ModularFormCoordinates.model_validate_json(left.model_dump_json())
    different = _form(2, 4, (2, 4))

    assert modular_form_coordinates_equal(left, same)
    assert not modular_form_coordinates_equal(left, different)


def test_catalog_equality_operation_returns_an_ordinary_boolean() -> None:
    tool = next(
        tool for tool in TOOLS if tool.operation_id == "modular_form.equal.check"
    )
    left = _form(2, 4, (2, 3))
    result = tool.run(tool.request_type(left=left, right=left))

    assert result.equal is True


def test_global_equality_rejects_different_exact_spaces() -> None:
    left = _form(1, 12, (1, 0))
    right = _form(2, 4, (2, 3))

    with pytest.raises(OperationDomainValidationError, match="equal weights"):
        modular_form_coordinates_equal(left, right)


def test_global_equality_compares_forms_across_nested_levels() -> None:
    level_one_e4 = _form(1, 4, (1,))
    level_two_e4 = ModularFormCoordinates(
        space=ModularFormSpace(level=2, weight=4, kind="M"),
        basis_id="gamma0-two-weight-2-4-monomials-v1",
        coordinates=(CanonicalRational(num=0, den=1), CanonicalRational(num=1, den=1)),
    )
    level_two_a2_squared = ModularFormCoordinates(
        space=level_two_e4.space,
        basis_id=level_two_e4.basis_id,
        coordinates=(CanonicalRational(num=1, den=1), CanonicalRational(num=0, den=1)),
    )

    assert modular_form_coordinates_equal(level_one_e4, level_two_e4)
    assert not modular_form_coordinates_equal(level_one_e4, level_two_a2_squared)


def test_global_equality_includes_cuspidal_forms_in_ambient_space() -> None:
    delta = ModularFormCoordinates(
        space=ModularFormSpace(level=1, weight=12, kind="S"),
        basis_id="level-one-e4-e6-monomials-v1",
        coordinates=(CanonicalRational(num=1, den=1),),
    )
    ambient_expression = ModularFormCoordinates(
        space=ModularFormSpace(level=1, weight=12, kind="M"),
        basis_id="level-one-e4-e6-monomials-v1",
        coordinates=(
            CanonicalRational(num=1, den=1728),
            CanonicalRational(num=-1, den=1728),
        ),
    )

    assert modular_form_coordinates_equal(delta, ambient_expression)


def test_global_equality_compares_forms_from_nonnested_levels() -> None:
    level_two_e4 = ModularFormCoordinates(
        space=ModularFormSpace(level=2, weight=4, kind="M"),
        basis_id="gamma0-two-weight-2-4-monomials-v1",
        coordinates=(CanonicalRational(num=0, den=1), CanonicalRational(num=1, den=1)),
    )
    level_three_e4 = ModularFormCoordinates(
        space=ModularFormSpace(level=3, weight=4, kind="M"),
        basis_id="gamma0-three-weight-2-4-6-hypersurface-v1",
        coordinates=(CanonicalRational(num=5, den=1), CanonicalRational(num=-4, den=1)),
    )

    assert modular_form_coordinates_equal(level_two_e4, level_three_e4)


def test_global_equality_re_admits_model_constructed_values() -> None:
    valid = _form(2, 4, (1, 0))
    forged = ModularFormCoordinates.model_construct(
        space=valid.space,
        basis_id=valid.basis_id,
        coordinates=(CanonicalRational(num=1, den=2),),
    )

    with pytest.raises(OperationDomainValidationError, match="coordinate count"):
        modular_form_coordinates_equal(valid, forged)


def test_global_equality_rejects_noncanonical_forged_rationals() -> None:
    valid = _form(2, 4, (1, 0))
    noncanonical = CanonicalRational.model_construct(num=2, den=2)
    forged = ModularFormCoordinates.model_construct(
        space=valid.space,
        basis_id=valid.basis_id,
        coordinates=(noncanonical, CanonicalRational(num=0, den=1)),
    )

    with pytest.raises(OperationDomainValidationError, match="reduced and canonical"):
        modular_form_coordinates_equal(valid, forged)


def test_global_equality_rejects_missing_forged_space() -> None:
    valid = _form(2, 4, (1, 0))
    forged = ModularFormCoordinates.model_construct(
        space=None,
        basis_id=valid.basis_id,
        coordinates=valid.coordinates,
    )

    with pytest.raises(
        OperationDomainValidationError, match="canonical modular-form space"
    ):
        modular_form_coordinates_equal(forged, valid)


def test_zero_dimensional_forms_have_the_unique_empty_coordinate_vector() -> None:
    space = ModularFormSpace(level=1, weight=0, kind="S")
    empty = ModularFormCoordinates(
        space=space,
        basis_id="level-one-e4-e6-monomials-v1",
        coordinates=(),
    )

    assert modular_form_coordinates_equal(empty, empty)
