"""Exact character transport across supplied finite unit bases."""

import json
import math

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.characters.coordinate_basis import (
    change_dirichlet_character_coordinate_basis as exported_coordinate_basis_change,
)
from jacobian.math.number_theory.characters.coordinate_basis._models import (
    DirichletCharacterBasisChangeRequest,
    DirichletCharacterBasisChangeResult,
    DirichletCharacterCoordinateIsomorphism,
)
from jacobian.math.number_theory.characters.coordinate_basis._tools import TOOLS
from jacobian.math.number_theory.characters.coordinate_basis.operations import (
    change_dirichlet_character_coordinate_basis,
)
from jacobian.math.number_theory.characters.operations import (
    dirichlet_character_conjugate,
    dirichlet_character_product,
    dirichlet_character_table,
    dirichlet_character_value,
    require_complete_character_group,
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


def test_operation_is_exported_from_its_owner_package() -> None:
    assert (
        exported_coordinate_basis_change is change_dirichlet_character_coordinate_basis
    )


def test_transported_character_composes_with_existing_consumers() -> None:
    request = _request()

    result = change_dirichlet_character_coordinate_basis(
        request.character, request.coordinate_isomorphism
    )
    transported = result.transported_character

    require_complete_character_group(transported.group)
    assert dirichlet_character_table(transported).values == result.values
    for residue in range(8):
        assert dirichlet_character_value(
            transported, residue
        ).value == _direct_character_value(request.character, residue)
    square = dirichlet_character_product(transported, transported)
    assert square.group == transported.group
    assert square.coordinates == (0, 0)
    conjugate = dirichlet_character_conjugate(transported)
    assert dirichlet_character_table(conjugate).values == tuple(
        None if value is None else value.conjugate() for value in result.values
    )


def test_transported_character_survives_serialization_into_consumers() -> None:
    request = _request()
    result = change_dirichlet_character_coordinate_basis(
        request.character, request.coordinate_isomorphism
    )

    decoded = DirichletCharacterBasisChangeResult.model_validate_json(
        result.model_dump_json()
    )

    require_complete_character_group(decoded.transported_character.group)
    assert (
        dirichlet_character_table(decoded.transported_character).values == result.values
    )


def test_group_admission_rejects_nonunit_order_one_generators() -> None:
    group = DirichletCharacterGroup(
        modulus=2,
        unit_residues=(1,),
        character_count=1,
        invariant_factors=(),
        generators=(0,),
        generator_orders=(1,),
        unit_coordinates=((0,),),
        exponent=1,
    )
    character = DirichletCharacter(group=group, coordinates=(0,))
    isomorphism = DirichletCharacterCoordinateIsomorphism(
        source_group=group,
        target_group=group,
        target_generator_images_in_source_coordinates=((0,),),
    )

    with pytest.raises(OperationDomainValidationError, match="exact") as error:
        change_dirichlet_character_coordinate_basis(character, isomorphism)
    assert error.value.errors()[0]["type"] == (
        "dirichlet_character.group.generator_order"
    )


def test_native_transport_rejects_wrong_runtime_argument_types() -> None:
    request = _request()

    with pytest.raises(OperationDomainValidationError, match="coordinate-basis"):
        change_dirichlet_character_coordinate_basis(
            request.character,
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(OperationDomainValidationError, match="Dirichlet character"):
        change_dirichlet_character_coordinate_basis(
            object(),  # type: ignore[arg-type]
            request.coordinate_isomorphism,
        )


def test_native_transport_rejects_a_shape_bypassed_isomorphism() -> None:
    request = _request()
    invalid = request.coordinate_isomorphism.model_copy(
        update={"target_generator_images_in_source_coordinates": ()}
    )

    with pytest.raises(OperationDomainValidationError, match="malformed") as error:
        change_dirichlet_character_coordinate_basis(request.character, invalid)
    assert error.value.errors()[0]["type"] == (
        "dirichlet_character.coordinate_basis.isomorphism_invalid"
    )
