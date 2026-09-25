"""Exact character transport across supplied finite unit bases."""

import json
import math

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.characters.coordinate_basis._models import (
    DirichletCharacterBasisChangeRequest,
    DirichletCharacterCoordinateIsomorphism,
)
from jacobian.math.number_theory.characters.coordinate_basis._tools import TOOLS
from jacobian.math.number_theory.characters.coordinate_basis.operations import (
    change_dirichlet_character_coordinate_basis,
)
from jacobian.math.number_theory.characters.values import (
    CyclotomicValue,
    DirichletCharacter,
    DirichletCharacterGroup,
)


def _group(generators: tuple[int, int]) -> DirichletCharacterGroup:
    if generators == (3, 5):
        rows = ((0, 0), (1, 0), (0, 1), (1, 1))
    else:
        rows = ((0, 0), (0, 1), (1, 0), (1, 1))
    return DirichletCharacterGroup(
        modulus=8,
        unit_residues=(1, 3, 5, 7),
        character_count=4,
        invariant_factors=(2, 2),
        generators=generators,
        generator_orders=(2, 2),
        unit_coordinates=rows,
        exponent=2,
    )


def _request() -> DirichletCharacterBasisChangeRequest:
    source = _group((3, 5))
    target = _group((5, 3))
    return DirichletCharacterBasisChangeRequest(
        character=DirichletCharacter(group=source, coordinates=(1, 0)),
        coordinate_isomorphism=DirichletCharacterCoordinateIsomorphism(
            source_group=source,
            target_group=target,
            target_generator_images_in_source_coordinates=((0, 1), (1, 0)),
        ),
    )


def _direct_character_value(
    character: DirichletCharacter, residue: int
) -> CyclotomicValue | None:
    group = character.group
    if math.gcd(residue, group.modulus) != 1:
        return None
    row = group.unit_coordinates[group.unit_residues.index(residue % group.modulus)]
    exponent = (
        sum(
            character_coordinate * (group.exponent // generator_order) * unit_coordinate
            for character_coordinate, generator_order, unit_coordinate in zip(
                character.coordinates, group.generator_orders, row, strict=True
            )
        )
        % group.exponent
    )
    return CyclotomicValue(order=group.exponent, exponent=exponent)


def test_basis_transport_preserves_independently_evaluated_residue_values() -> None:
    request = _request()

    result = change_dirichlet_character_coordinate_basis(
        request.character, request.coordinate_isomorphism
    )

    assert result.transported_character.coordinates == (0, 1)
    assert result.residues == tuple(range(8))
    assert result.values == tuple(
        _direct_character_value(request.character, residue) for residue in range(8)
    )
    assert result.model_validate_json(result.model_dump_json()) == result


def test_public_operation_executes_complete_transport_example() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "dirichlet_character.change_coordinate_basis.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))

    result = tool.run(request)

    assert result.transported_character.coordinates == (0, 1)
    assert result.values[0] is None
    assert result.values[1:] == (
        CyclotomicValue(order=2, exponent=0),
        None,
        CyclotomicValue(order=2, exponent=1),
        None,
        CyclotomicValue(order=2, exponent=0),
        None,
        CyclotomicValue(order=2, exponent=1),
    )


def test_transport_rejects_a_map_that_does_not_identify_canonical_units() -> None:
    request = _request()
    invalid = request.coordinate_isomorphism.model_copy(
        update={"target_generator_images_in_source_coordinates": ((1, 0), (0, 1))}
    )

    with pytest.raises(OperationDomainValidationError, match="same unit"):
        change_dirichlet_character_coordinate_basis(request.character, invalid)


def test_transport_rejects_a_structurally_shaped_but_false_unit_decomposition() -> None:
    request = _request()
    bad_target = request.coordinate_isomorphism.target_group.model_copy(
        update={"unit_coordinates": ((0, 0), (1, 0), (0, 1), (1, 1))}
    )
    invalid = request.coordinate_isomorphism.model_copy(
        update={"target_group": bad_target}
    )

    with pytest.raises(OperationDomainValidationError, match="reconstruct"):
        change_dirichlet_character_coordinate_basis(request.character, invalid)


def test_isomorphism_input_bounds_generator_image_rank() -> None:
    request = _request()
    with pytest.raises(ValidationError, match="target-generator image"):
        DirichletCharacterCoordinateIsomorphism(
            source_group=request.character.group,
            target_group=request.coordinate_isomorphism.target_group,
            target_generator_images_in_source_coordinates=((0,), (1,)),
        )


def test_trivial_unit_group_transport_preserves_its_degenerate_table() -> None:
    group = DirichletCharacterGroup(
        modulus=1,
        unit_residues=(0,),
        character_count=1,
        invariant_factors=(),
        generators=(),
        generator_orders=(),
        unit_coordinates=((),),
        exponent=1,
    )
    character = DirichletCharacter(group=group, coordinates=())
    isomorphism = DirichletCharacterCoordinateIsomorphism(
        source_group=group,
        target_group=group,
        target_generator_images_in_source_coordinates=(),
    )

    result = change_dirichlet_character_coordinate_basis(character, isomorphism)

    assert result.transported_character.coordinates == ()
    assert result.residues == (0,)
    assert result.values == (CyclotomicValue(order=1, exponent=0),)
