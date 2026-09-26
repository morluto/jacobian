"""Public Sturm-bound contract checked against the exact Gamma0 index formula."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.cyclic_linear._models import RationalCyclotomicField
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
)
from jacobian.math.number_theory.characters.values import (
    MAX_CHARACTER_GROUP_MODULUS,
    DirichletCharacter,
)
from jacobian.math.number_theory.modular_forms.transform_models import (
    SturmBoundRequest,
    SturmBoundResult,
)
from jacobian.math.number_theory.modular_forms.transform_tools import TOOLS
from jacobian.math.number_theory.modular_forms.transforms import sturm_bound
from jacobian.math.number_theory.modular_forms.values import ModularFormSpace

_STURM_TOOL = next(
    tool
    for tool in TOOLS
    if tool.operation_id == "modular_form.space.sturm_bound.compute"
)


def _space(level: int, weight: int, kind: str = "M") -> ModularFormSpace:
    return ModularFormSpace(
        level=level,
        weight=weight,
        kind=kind,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("level", "weight", "index", "bound", "determining_terms"),
    [
        # A bound B determines coefficients at indices 0 through B, or B+1 terms.
        (1, 12, 1, 1, 2),
        (11, 2, 12, 2, 3),
        (144, 3, 288, 72, 73),
    ],
)
def test_public_sturm_operation_matches_exact_gamma0_convention(
    level: int,
    weight: int,
    index: int,
    bound: int,
    determining_terms: int,
) -> None:
    """Check exact examples using an independent prime-factor index product."""

    expected_index = level
    for prime in {2, 3, 5, 7, 11}:
        if level % prime == 0:
            expected_index = expected_index // prime * (prime + 1)
    assert expected_index == index
    assert determining_terms == bound + 1

    request = SturmBoundRequest(space=_space(level, weight).model_dump())
    result = _STURM_TOOL.run(request)

    assert result.space == request.space
    assert result.index == index
    assert result.bound == bound


def test_public_sturm_operation_preserves_supported_character_parent() -> None:
    space = ModularFormSpace(
        level=4,
        weight=3,
        kind="M",
        character={
            "group": {
                "modulus": 4,
                "unit_residues": [1, 3],
                "character_count": 2,
                "invariant_factors": [2],
                "generators": [3],
                "generator_orders": [2],
                "unit_coordinates": [[0], [1]],
                "exponent": 2,
            },
            "coordinates": [1],
        },
    )
    request = SturmBoundRequest(space=space)
    result = _STURM_TOOL.run(request)

    assert result.space == space
    assert result.index == 6
    assert result.bound == 1


@pytest.mark.parametrize(
    ("level", "weight", "character_coordinates", "field_order", "kind"),
    [
        (13, 0, (4,), 6, "M"),  # zero space bound remains the exact integer zero
        (13, 2, (4,), 6, "M"),  # order-three character in Q(zeta_6)
        (13, 2, (3,), 12, "S"),  # order-four character in Q(zeta_12)
        (39, 4, (0, 4), 12, "M"),  # inflated-family parent at another level
    ],
)
def test_sturm_bound_covers_every_exact_coefficient_parent(
    level: int,
    weight: int,
    character_coordinates: tuple[int, ...],
    field_order: int,
    kind: str,
) -> None:
    group = character_group(level)
    character = dirichlet_character(group, character_coordinates)
    space = ModularFormSpace(
        level=level,
        weight=weight,
        kind=kind,  # type: ignore[arg-type]
        character=character,
        coefficient_domain=RationalCyclotomicField(order=field_order),
    )
    result = _STURM_TOOL.run(SturmBoundRequest(space=space))
    expected_index = level
    prime = 2
    remaining = level
    while prime * prime <= remaining:
        if remaining % prime == 0:
            expected_index = expected_index // prime * (prime + 1)
            while remaining % prime == 0:
                remaining //= prime
        prime += 1
    if remaining > 1:
        expected_index = expected_index // remaining * (remaining + 1)

    assert result.space == space
    assert result.index == expected_index
    assert result.bound == weight * expected_index // 12
    assert SturmBoundResult.model_validate(result.model_dump(mode="json")) == result


def test_sturm_rejects_fabricated_character_group_claim() -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    character = dirichlet_character(character_group(13), (2,))
    forged_group = character.group.model_copy(update={"generator_orders": (6,)})
    forged_character = character.model_copy(update={"group": forged_group})
    space = ModularFormSpace(
        level=13,
        weight=2,
        kind="M",
        character=forged_character,
        coefficient_domain=RationalCyclotomicField(order=6),
    )
    with pytest.raises(OperationDomainValidationError):
        sturm_bound(space)


def test_sturm_operation_accepts_cyclotomic_extension_of_trivial_space() -> None:
    space = ModularFormSpace(
        level=1,
        weight=12,
        kind="M",
        coefficient_domain=RationalCyclotomicField(order=3),
    )
    result = sturm_bound(space)
    assert result.space == space
    assert result.bound == 1


def test_order_four_character_example_runs_through_typed_catalog_contract() -> None:
    example = next(
        example
        for example in _STURM_TOOL.examples
        if example.name == "order_four_character_sturm"
    )
    request = SturmBoundRequest.model_validate(example.input)
    result = _STURM_TOOL.run(request)
    assert result.space == request.space
    assert result.index == 14
    assert result.bound == 2


def test_sturm_operation_admits_arithmetic_before_computing_bound() -> None:
    too_large_level = _space(10_001, 12)
    with pytest.raises(OperationResourceAdmissionError):
        _STURM_TOOL.run(SturmBoundRequest(space=too_large_level))


def test_sturm_bounds_constructed_character_tables_before_copying() -> None:
    valid = dirichlet_character(character_group(13), (2,))
    fields = valid.group.model_dump()
    fields["unit_residues"] = valid.group.unit_residues + (0,) * (
        MAX_CHARACTER_GROUP_MODULUS + 1
    )
    oversized_group = type(valid.group).model_construct(**fields)
    forged_character = DirichletCharacter.model_construct(
        group=oversized_group,
        coordinates=valid.coordinates,
    )
    space = ModularFormSpace.model_construct(
        group="GAMMA0",
        level=13,
        weight=2,
        kind="M",
        character=forged_character,
        coefficient_domain=RationalCyclotomicField(order=6),
    )

    with pytest.raises(OperationResourceAdmissionError):
        sturm_bound(space)
