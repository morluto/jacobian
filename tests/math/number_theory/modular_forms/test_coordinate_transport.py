from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.modular_forms import (
    ModularFormCoordinates,
    ModularFormSpace,
    ModularFormSpaceInclusion,
    basis,
    modular_form_coordinates_q_expansion,
    modular_form_coordinates_transport,
    modular_form_space_inclusion,
)


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
    inclusion = modular_form_space_inclusion(source.space, target)
    transported = modular_form_coordinates_transport(source, inclusion)

    assert transported.space == target
    assert transported.basis_id == "gamma0-two-weight-2-4-monomials-v1"
    transported_values = []
    for value in transported.coordinates:
        assert isinstance(value, CanonicalRational)
        transported_values.append(value.as_fraction())
    assert transported_values == [Fraction(0), Fraction(1)]

    expansion = modular_form_coordinates_q_expansion(transported, 8)
    # E4 = 1 + 240 * sum_{n>=1} sigma_3(n) q^n, evaluated independently.
    expected = [Fraction(1)] + [
        Fraction(240 * sum(d**3 for d in range(1, n + 1) if n % d == 0))
        for n in range(1, 8)
    ]
    coefficients = []
    for value in expansion.q_expansion.coefficients:
        assert isinstance(value, CanonicalRational)
        coefficients.append(value.as_fraction())
    assert coefficients == expected

    same = ModularFormCoordinates.model_validate_json(transported.model_dump_json())
    assert same == transported


def test_identical_pari_space_transport_preserves_coordinates_without_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = ModularFormSpace(level=5, weight=4, kind="M")
    form = _coordinates(5, 4, "M", "gamma0-rational-gamma0-sturm-rref-v1", (3, -1, 2))
    calls: list[object] = []
    materialize = basis._materialize_pari_basis

    def counting(plan: basis._BasisPlan) -> basis._BasisPlan:
        calls.append(plan)
        return materialize(plan)

    monkeypatch.setattr(basis, "_materialize_pari_basis", counting)
    inclusion = modular_form_space_inclusion(space, space)
    assert modular_form_coordinates_transport(form, inclusion) is form
    assert calls == []


def test_transport_cusp_form_into_ambient_space_preserves_the_form() -> None:
    delta = _coordinates(1, 12, "S", "level-one-e4-e6-monomials-v1", (1,))
    target = ModularFormSpace(level=2, weight=12, kind="M")
    inclusion = modular_form_space_inclusion(delta.space, target)
    transported = modular_form_coordinates_transport(delta, inclusion)
    expansion = modular_form_coordinates_q_expansion(delta, 8)
    transported_expansion = modular_form_coordinates_q_expansion(transported, 8)

    assert [
        c.as_fraction() for c in transported_expansion.q_expansion.coefficients
    ] == [c.as_fraction() for c in expansion.q_expansion.coefficients]


def test_transport_rejects_non_nested_parent_requests() -> None:
    form = _coordinates(2, 4, "M", "gamma0-two-weight-2-4-monomials-v1", (0, 1))
    with pytest.raises(OperationDomainValidationError) as exc_info:
        modular_form_space_inclusion(
            form.space, ModularFormSpace(level=3, weight=4, kind="M")
        )
    assert (
        exc_info.value.errors()[0]["type"] == "modular_form.inclusion_inclusion_level"
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        modular_form_space_inclusion(
            form.space, ModularFormSpace(level=4, weight=6, kind="M")
        )
    assert (
        exc_info.value.errors()[0]["type"] == "modular_form.inclusion_inclusion_weight"
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        modular_form_space_inclusion(
            form.space, ModularFormSpace(level=4, weight=4, kind="S")
        )
    assert exc_info.value.errors()[0]["type"] == "modular_form.inclusion_inclusion_kind"


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
    with pytest.raises(OperationDomainValidationError) as exc_info:
        modular_form_space_inclusion(source.space, target)
    assert (
        exc_info.value.errors()[0]["type"] == "modular_form.inclusion_inclusion_parent"
    )


def test_transport_rejects_forged_inclusion_and_source_mismatch() -> None:
    source = _coordinates(1, 4, "M", "level-one-e4-e6-monomials-v1", (1,))
    target = ModularFormSpace(level=2, weight=4, kind="M")
    forged = ModularFormSpaceInclusion.model_construct(
        map_kind="natural_gamma0_level_inclusion",
        source_space=ModularFormSpace(level=2, weight=4, kind="M"),
        target_space=ModularFormSpace(level=3, weight=4, kind="M"),
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        modular_form_coordinates_transport(source, forged)
    assert (
        exc_info.value.errors()[0]["type"]
        == "modular_form.transport_inclusion_inclusion_level"
    )

    other_source = ModularFormSpace(level=2, weight=4, kind="M")
    inclusion = modular_form_space_inclusion(other_source, target)
    with pytest.raises(OperationDomainValidationError) as exc_info:
        modular_form_coordinates_transport(source, inclusion)
    assert (
        exc_info.value.errors()[0]["type"] == "modular_form.transport_source_mismatch"
    )
