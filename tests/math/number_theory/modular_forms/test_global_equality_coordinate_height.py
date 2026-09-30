"""Global equality must bound the cyclotomic product height, not only the total."""

from __future__ import annotations

from itertools import product
from math import gcd
from typing import Any, Literal, cast

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_value,
)
from jacobian.math.number_theory.modular_forms.character_coordinates import (
    _admit_coordinate_space,
)
from jacobian.math.number_theory.modular_forms.global_equality import (
    CyclotomicFieldEmbedding,
)
from jacobian.math.number_theory.modular_forms.global_equality.operations import (
    modular_form_coordinates_global_equal,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)

_SOURCE_FIELD = RationalCyclotomicField(order=6)
_TARGET_FIELD = RationalCyclotomicField(order=12)


def _inflated_character(level: int, coordinate: int) -> Any:
    """The level-``level`` character whose values match one of conductor 13."""
    source = dirichlet_character(character_group(13), (coordinate,))
    target_group = character_group(level)
    for coordinates in product(
        *(range(order) for order in target_group.generator_orders)
    ):
        candidate = dirichlet_character(target_group, coordinates)
        if all(
            dirichlet_character_value(candidate, residue).value
            == dirichlet_character_value(source, residue).value
            for residue in range(level)
            if gcd(residue, level) == 1
        ):
            return candidate
    raise AssertionError("character inflation fixture was not found")


def _space(level: int, coordinate: int, kind: Literal["M", "S"]) -> ModularFormSpace:
    return ModularFormSpace(
        level=level,
        weight=2,
        kind=kind,
        character=_inflated_character(level, coordinate),
        coefficient_domain=_SOURCE_FIELD,
    )


def _embedding() -> CyclotomicFieldEmbedding:
    # The inclusion Q(zeta_6) -> Q(zeta_12) sends zeta_6 to zeta_12^2.
    return CyclotomicFieldEmbedding(
        source_order=6,
        target_field=_TARGET_FIELD,
        generator_image=RationalCyclotomicElement(
            field=_TARGET_FIELD,
            coefficients_ascending=tuple(
                CanonicalRational(num=int(index == 2), den=1) for index in range(4)
            ),
        ),
    )


def _conjugate_embedding() -> CyclotomicFieldEmbedding:
    # zeta_6 maps to zeta_12^-2 = 1 - zeta_12^2.
    return CyclotomicFieldEmbedding(
        source_order=6,
        target_field=_TARGET_FIELD,
        generator_image=RationalCyclotomicElement(
            field=_TARGET_FIELD,
            coefficients_ascending=tuple(
                CanonicalRational(
                    num=(1 if index == 0 else -1 if index == 2 else 0), den=1
                )
                for index in range(4)
            ),
        ),
    )


def _coordinates(space: ModularFormSpace, value: int) -> ModularFormCoordinates:
    return ModularFormCoordinates(
        space=space,
        basis_id=cast(Any, _admit_coordinate_space(space).basis_id),
        coordinates=(
            RationalCyclotomicElement(
                field=_SOURCE_FIELD,
                coefficients_ascending=(
                    CanonicalRational(num=value, den=1),
                    CanonicalRational(num=0, den=1),
                ),
            ),
        ),
    )


def test_a_coordinate_too_wide_to_multiply_is_refused_at_admission() -> None:
    """The product height grows multiplicatively; the budget grew linearly.

    ``intermediate_digits`` in this operation's admission grows *linearly* in
    coordinate height, while the Sturm prefix multiplies degree-``d`` cyclotomic
    elements and one product reaches ``(2d + 2) * operand_digits +
    len(str(d)) + 2`` digits. For a Q(zeta_12) target the two rules part company
    at 26 digits -- inside this budget's cap, already past
    ``MAX_CYCLIC_FIELD_ELEMENT_DIGITS`` -- so a deterministically unsupported
    request used to launch the Sturm worker and then be refused by the backend.
    """
    assert _TARGET_FIELD.degree == 4
    wide = 10**25
    assert len(str(wide)) == 26
    assert wide * 10 + 3 > MAX_CYCLIC_FIELD_ELEMENT_DIGITS

    left = _coordinates(_space(13, 2, "S"), wide)
    right = _coordinates(_space(13, 10, "S"), 1)

    with pytest.raises(OperationResourceAdmissionError) as refusal:
        modular_form_coordinates_global_equal(
            left, _embedding(), right, _conjugate_embedding()
        )
    assert (
        refusal.value.errors()[0]["type"] == "modular_form.global_equality_output_bound"
    )


def test_a_narrow_coordinate_of_the_same_shape_is_unaffected() -> None:
    """Negative control: the added charge must not refuse ordinary input."""
    left = _coordinates(_space(13, 2, "S"), 2)
    right = _coordinates(_space(13, 10, "S"), 2)

    try:
        result = modular_form_coordinates_global_equal(
            left, _embedding(), right, _conjugate_embedding()
        )
    except OperationResourceAdmissionError as refusal:
        assert (
            refusal.errors()[0]["type"] != "modular_form.global_equality_output_bound"
        )
    else:
        assert isinstance(result, bool)
