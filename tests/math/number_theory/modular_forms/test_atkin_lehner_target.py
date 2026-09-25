import pytest

from jacobian.math.matrices.cyclic_linear._models import RationalCyclotomicField
from jacobian.math.number_theory.characters.operations import dirichlet_character_value
from jacobian.math.number_theory.characters.values import (
    DirichletCharacter,
    DirichletCharacterGroup,
)
from jacobian.math.number_theory.modular_forms._tools import TOOLS
from jacobian.math.number_theory.modular_forms.operations import (
    modular_form_atkin_lehner_target,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormAtkinLehnerTarget,
    ModularFormSpace,
)


def _order_four_character() -> DirichletCharacter:
    group = DirichletCharacterGroup(
        modulus=5,
        unit_residues=(1, 2, 3, 4),
        character_count=4,
        invariant_factors=(4,),
        generators=(2,),
        generator_orders=(4,),
        unit_coordinates=((0,), (1,), (3,), (2,)),
        exponent=4,
    )
    return DirichletCharacter(group=group, coordinates=(1,))


def test_full_fricke_target_inverts_character_on_every_unit() -> None:
    character = _order_four_character()
    source = ModularFormSpace(
        level=5,
        weight=4,
        kind="S",
        character=character,
        coefficient_domain=RationalCyclotomicField(order=4),
    )

    result = modular_form_atkin_lehner_target(source)
    restored = ModularFormAtkinLehnerTarget.model_validate_json(
        result.model_dump_json()
    )

    assert restored == result
    assert result.divisor == source.level
    assert result.target_space.level == source.level
    assert result.target_space.weight == source.weight
    assert result.target_space.kind == source.kind
    assert result.target_space.coefficient_domain == source.coefficient_domain
    assert result.target_space.character.coordinates == (3,)
    for unit in character.group.unit_residues:
        source_value = dirichlet_character_value(character, unit).value
        target_value = dirichlet_character_value(
            result.target_space.character, unit
        ).value
        assert source_value is not None and target_value is not None
        assert (source_value.exponent + target_value.exponent) % 4 == 0


def test_trivial_character_fricke_target_is_same_parent() -> None:
    source = ModularFormSpace(level=8, weight=2, kind="M")
    result = modular_form_atkin_lehner_target(source)
    assert result.target_space == source


def test_catalog_declares_full_fricke_target_operation() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "modular_form.atkin_lehner.target_space.compute"
    )
    request = tool.request_type.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert result.target_space.character.coordinates == (3,)


def test_target_carrier_rejects_different_weight() -> None:
    source = ModularFormSpace(level=5, weight=4, kind="M")
    target = ModularFormSpace(level=5, weight=6, kind="M")
    with pytest.raises(ValueError, match="preserves subgroup, level, weight"):
        ModularFormAtkinLehnerTarget(
            source_space=source,
            target_space=target,
            divisor=5,
        )


def test_target_carrier_rejects_partial_divisor() -> None:
    source = ModularFormSpace(level=5, weight=4, kind="M")
    with pytest.raises(ValueError, match="full Fricke divisor"):
        ModularFormAtkinLehnerTarget(
            source_space=source,
            target_space=source,
            divisor=1,
        )
