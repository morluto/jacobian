from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.modular_forms import (
    ModularFormCoordinates,
    ModularFormSpace,
    modular_form_coordinates_q_expansion,
    modular_form_coordinates_transport,
)
from jacobian.math.number_theory.modular_forms._tools import TOOLS


def _coordinates(
    level: int, weight: int, kind: str, basis_id: str, values: tuple[int, ...]
) -> ModularFormCoordinates:
    return ModularFormCoordinates(
        space=ModularFormSpace(level=level, weight=weight, kind=kind),  # type: ignore[arg-type]
        basis_id=basis_id,  # type: ignore[arg-type]
        coordinates=tuple(CanonicalRational(num=value, den=1) for value in values),
    )


def test_transport_level_one_e4_to_gamma0_two_matches_independent_divisor_sum() -> None:
    source = _coordinates(1, 4, "M", "level-one-e4-e6-monomials-v1", (1,))
    target = ModularFormSpace(level=2, weight=4, kind="M")
    transported = modular_form_coordinates_transport(source, target)

    assert transported.space == target
    assert transported.basis_id == "gamma0-two-weight-2-4-monomials-v1"
    assert tuple(value.as_fraction() for value in transported.coordinates) == (
        Fraction(0),
        Fraction(1),
    )

    expansion = modular_form_coordinates_q_expansion(transported, 8)
    # E4 = 1 + 240 * sum_{n>=1} sigma_3(n) q^n, evaluated independently.
    expected = [Fraction(1)] + [
        Fraction(240 * sum(d**3 for d in range(1, n + 1) if n % d == 0))
        for n in range(1, 8)
    ]
    assert [
        value.as_fraction() for value in expansion.q_expansion.coefficients
    ] == expected

    same = ModularFormCoordinates.model_validate_json(transported.model_dump_json())
    assert same == transported


def test_transport_cusp_form_into_ambient_space_preserves_the_form() -> None:
    delta = _coordinates(1, 12, "S", "level-one-e4-e6-monomials-v1", (1,))
    target = ModularFormSpace(level=2, weight=12, kind="M")
    transported = modular_form_coordinates_transport(delta, target)
    expansion = modular_form_coordinates_q_expansion(delta, 8)
    transported_expansion = modular_form_coordinates_q_expansion(transported, 8)

    assert [
        c.as_fraction() for c in transported_expansion.q_expansion.coefficients
    ] == [c.as_fraction() for c in expansion.q_expansion.coefficients]


def test_transport_rejects_non_nested_parent_requests() -> None:
    form = _coordinates(2, 4, "M", "gamma0-two-weight-2-4-monomials-v1", (0, 1))
    with pytest.raises(OperationDomainValidationError, match="divide"):
        modular_form_coordinates_transport(
            form, ModularFormSpace(level=3, weight=4, kind="M")
        )
    with pytest.raises(OperationDomainValidationError, match="weights must agree"):
        modular_form_coordinates_transport(
            form, ModularFormSpace(level=4, weight=6, kind="M")
        )
    with pytest.raises(OperationDomainValidationError, match="does not embed"):
        modular_form_coordinates_transport(
            form, ModularFormSpace(level=4, weight=4, kind="S")
        )


def test_transport_rejects_nontrivial_character_target() -> None:
    from jacobian.math.number_theory.characters.values import (
        DirichletCharacter,
        DirichletCharacterGroup,
    )

    source = _coordinates(2, 4, "M", "gamma0-two-weight-2-4-monomials-v1", (0, 1))
    group = DirichletCharacterGroup(
        modulus=4,
        unit_residues=(1, 3),
        character_count=2,
        invariant_factors=(2,),
        generators=(3,),
        generator_orders=(2,),
        unit_coordinates=((0,), (1,)),
        exponent=2,
    )
    character = DirichletCharacter(group=group, coordinates=(1,))
    target = ModularFormSpace(level=4, weight=4, kind="M", character=character)
    with pytest.raises(OperationDomainValidationError, match="QQ trivial-character"):
        modular_form_coordinates_transport(source, target)


def test_transport_catalog_operation_round_trips_target_coordinates() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "modular_form.coordinates.transport.compute"
    )
    source = _coordinates(1, 4, "M", "level-one-e4-e6-monomials-v1", (1,))
    request = tool.request_type(
        form=source, target_space=ModularFormSpace(level=2, weight=4, kind="M")
    )
    result = tool.run(request)

    assert result.space.level == 2
    assert (
        ModularFormCoordinates.model_validate_json(result.model_dump_json()) == result
    )
