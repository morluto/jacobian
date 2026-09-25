"""Canonical modular-space coefficient-field representation tests."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms.basis import (
    modular_form_basis_q_expansions,
)
from jacobian.math.number_theory.modular_forms.character_basis import (
    modular_character_coordinates_q_expansion,
)
from jacobian.math.number_theory.modular_forms.operations import space_dimension
from jacobian.math.number_theory.modular_forms.transform_models import SturmBoundResult
from jacobian.math.number_theory.modular_forms.transforms import sturm_bound
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
    ModularQExpansion,
)
from jacobian.math.polynomials.series._models import TruncatedSeries


def _character_order_six() -> DirichletCharacter:
    # The mod-13 unit group is cyclic of order 12; its second power has exact
    # character order 6 and is even, so it is compatible with weight 2.
    return dirichlet_character(character_group(13), (2,))


def test_space_binds_nonrational_character_to_explicit_cyclotomic_field() -> None:
    space = ModularFormSpace(
        level=13,
        weight=2,
        kind="M",
        character=_character_order_six(),
        coefficient_domain=RationalCyclotomicField(order=6),
    )

    restored = TypeAdapter(ModularFormSpace).validate_json(space.model_dump_json())
    assert restored == space
    assert restored.character == space.character
    assert restored.coefficient_domain == RationalCyclotomicField(order=6)


def test_space_allows_explicit_cyclotomic_extension_containing_character_values() -> (
    None
):
    space = ModularFormSpace(
        level=13,
        weight=2,
        kind="S",
        character=_character_order_six(),
        coefficient_domain=RationalCyclotomicField(order=12),
    )

    assert space.coefficient_domain.order == 12


def test_space_rejects_nonrational_character_over_qq() -> None:
    with pytest.raises(ValidationError, match="QQ coefficients cannot contain"):
        ModularFormSpace(
            level=13,
            weight=2,
            kind="M",
            character=_character_order_six(),
            coefficient_domain="QQ",
        )


def test_space_rejects_coefficient_field_that_does_not_contain_character_values() -> (
    None
):
    with pytest.raises(ValidationError, match="must contain the character value field"):
        ModularFormSpace(
            level=13,
            weight=2,
            kind="M",
            character=_character_order_six(),
            coefficient_domain=RationalCyclotomicField(order=3),
        )


def test_space_defers_the_complete_group_proof_to_relying_operations() -> None:
    character = _character_order_six()
    forged_group = character.group.model_copy(update={"generator_orders": (6,)})
    forged_character = character.model_copy(update={"group": forged_group})

    space = ModularFormSpace(
        level=13,
        weight=2,
        kind="S",
        character=forged_character,
        coefficient_domain=RationalCyclotomicField(order=6),
    )

    field = RationalCyclotomicField(order=6)
    form = ModularFormCoordinates(
        space=space,
        basis_id="gamma0-13-even-order6-character-sturm-v1",
        coordinates=(
            RationalCyclotomicElement(
                field=field,
                coefficients_ascending=(
                    CanonicalRational(num=1, den=1),
                    CanonicalRational(num=0, den=1),
                ),
            ),
        ),
    )

    with pytest.raises(OperationDomainValidationError) as excinfo:
        modular_character_coordinates_q_expansion(form)
    assert excinfo.value.errors()[0]["type"].startswith("dirichlet_character.group")


def test_space_requires_explicit_character_inflation_to_the_level() -> None:
    character = dirichlet_character(character_group(13), (2,))
    with pytest.raises(ValidationError, match="must equal the Gamma0 level"):
        ModularFormSpace(
            level=26,
            weight=2,
            kind="M",
            character=character,
            coefficient_domain=RationalCyclotomicField(order=6),
        )


def test_space_bounds_cyclotomic_field_before_any_form_expansion() -> None:
    with pytest.raises(ValidationError, match="degree at most 32"):
        ModularFormSpace(
            level=13,
            weight=2,
            kind="M",
            character=_character_order_six(),
            coefficient_domain=RationalCyclotomicField(order=128),
        )


def test_existing_qq_operations_reject_the_new_parent_without_claiming_support() -> (
    None
):
    space = ModularFormSpace(
        level=13,
        weight=2,
        kind="M",
        character=_character_order_six(),
        coefficient_domain=RationalCyclotomicField(order=6),
    )

    for operation in (
        lambda: space_dimension(space),
        lambda: sturm_bound(space),
        lambda: modular_form_basis_q_expansions(space, 4),
    ):
        with pytest.raises(OperationDomainValidationError):
            operation()


def test_shared_coordinate_carrier_preserves_cyclotomic_parent_and_scalar_type() -> (
    None
):
    space = ModularFormSpace(
        level=13,
        weight=2,
        kind="M",
        character=_character_order_six(),
        coefficient_domain=RationalCyclotomicField(order=6),
    )

    coordinates = ModularFormCoordinates(
        space=space,
        basis_id="gamma0-13-even-order6-character-sturm-v1",
        coordinates=(
            RationalCyclotomicElement(
                field=space.coefficient_domain,
                coefficients_ascending=(
                    CanonicalRational(num=1, den=1),
                    CanonicalRational(num=0, den=1),
                ),
            ),
        ),
    )
    assert (
        ModularFormCoordinates.model_validate_json(coordinates.model_dump_json())
        == coordinates
    )

    with pytest.raises(ValidationError, match="must belong to the declared space"):
        ModularFormCoordinates(
            space=space,
            basis_id="gamma0-13-even-order6-character-sturm-v1",
            coordinates=(CanonicalRational(num=1, den=1),),
        )
    with pytest.raises(ValidationError, match="requires a QQ space"):
        ModularQExpansion(
            space=space,
            weight=2,
            q_expansion=TruncatedSeries(
                variable="q",
                truncation_order=1,
                coefficients=(CanonicalRational(num=0, den=1),),
            ),
        )


def test_rational_sturm_result_cannot_be_deserialized_for_cyclotomic_space() -> None:
    space = ModularFormSpace(
        level=13,
        weight=2,
        kind="M",
        character=_character_order_six(),
        coefficient_domain=RationalCyclotomicField(order=6),
    )

    with pytest.raises(ValidationError, match="represents QQ spaces only"):
        SturmBoundResult(space=space, index=14, bound=2)


def test_rational_character_values_remain_valid_over_qq() -> None:
    character = dirichlet_character(character_group(4), (1,))
    space = ModularFormSpace(
        level=4,
        weight=1,
        kind="M",
        character=character,
        coefficient_domain="QQ",
    )

    assert space.coefficient_domain == "QQ"
