from __future__ import annotations

import itertools
import json

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.groups.actions._models import FinitePermutationAction
from jacobian.math.groups.characters.operations import class_function_inner_product
from jacobian.math.groups.characters.permutation._models import FiniteCharacter
from jacobian.math.groups.characters.permutation._tools import TOOLS
from jacobian.math.groups.characters.permutation.operations import (
    PermutationCharacterRequest,
    permutation_character,
)


def _compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left[right[index]] for index in range(len(left)))


def _direct_s3_classes() -> tuple[tuple[tuple[int, ...], ...], ...]:
    elements = tuple(itertools.permutations(range(3)))
    unseen = set(elements)
    classes = []
    for representative in elements:
        if representative not in unseen:
            continue
        conjugates = {
            _compose(_compose(conjugator, representative), _inverse(conjugator))
            for conjugator in elements
        }
        unseen.difference_update(conjugates)
        classes.append(tuple(sorted(conjugates)))
    return tuple(sorted(classes, key=lambda row: row[0]))


def _inverse(value: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(value)
    for index, image in enumerate(value):
        result[image] = index
    return tuple(result)


def test_s3_fixed_point_values_and_classes_match_direct_permutation_oracle() -> None:
    action = FinitePermutationAction(
        domain=("a", "b", "c"),
        generators=((1, 0, 2), (1, 2, 0)),
    )
    result = permutation_character(PermutationCharacterRequest(action=action))

    classes = _direct_s3_classes()
    expected_values = tuple(
        sum(point == conjugacy_class[0][point] for point in range(3))
        for conjugacy_class in classes
    )
    assert result.partition.classes == classes
    assert (
        tuple(value.coefficients[0].num for value in result.values) == expected_values
    )
    assert expected_values == (3, 1, 0)
    assert result.axis.group == result.partition.source


def test_trivial_group_action_is_a_degree_three_permutation_character() -> None:
    action = FinitePermutationAction(
        domain=("red", "green", "blue"), generators=((0, 1, 2),)
    )
    result = permutation_character(PermutationCharacterRequest(action=action))

    assert result.partition.classes == (((0, 1, 2),),)
    assert result.values[0].coefficients[0].as_fraction() == 3
    assert result.action == action


def test_character_serialization_and_tampering_are_checked() -> None:
    action = FinitePermutationAction(
        domain=("a", "b", "c"), generators=((1, 0, 2), (1, 2, 0))
    )
    result = permutation_character(PermutationCharacterRequest(action=action))
    decoded = FiniteCharacter.model_validate_json(result.model_dump_json())
    assert decoded == result

    forged_value = result.model_dump(mode="json")
    forged_value["values"][0]["coefficients"][0]["num"] = "9"
    with pytest.raises(ValidationError, match="fixed-point counts"):
        FiniteCharacter.model_validate_json(json.dumps(forged_value))

    forged_partition = result.model_dump(mode="json")
    forged_partition["partition"]["classes"] = [
        [[0, 1, 2]],
        [[1, 0, 2]],
        [[1, 2, 0]],
    ]
    with pytest.raises(ValidationError, match="complete action-group partition"):
        FiniteCharacter.model_validate_json(json.dumps(forged_partition))

    oversized_partition = result.model_dump(mode="json")
    oversized_partition["partition"]["classes"] = [[]] * 65
    with pytest.raises(ValidationError, match="class-count bound"):
        FiniteCharacter.model_validate_json(json.dumps(oversized_partition))

    with pytest.raises(ValidationError, match="fixed-point counts"):
        result.model_copy(update={"values": tuple(reversed(result.values))})


def test_typed_constructed_action_and_axis_are_readmitted() -> None:
    action = FinitePermutationAction(domain=("a", "b"), generators=((1, 0),))
    result = permutation_character(PermutationCharacterRequest(action=action))
    forged_axis = result.axis.model_copy(update={"group_order": 999})
    forged_payload = result.model_dump(mode="python")
    forged_payload["axis"] = forged_axis
    with pytest.raises(ValidationError, match=r"group_order|axis must match"):
        FiniteCharacter.model_validate(forged_payload)

    oversized_action = FinitePermutationAction.model_construct(
        domain=tuple(f"p{i}" for i in range(51)),
        generators=(tuple(range(51)),),
    )
    with pytest.raises(ValidationError, match="structural bounds"):
        PermutationCharacterRequest.model_validate({"action": oversized_action})


def test_action_group_order_is_rejected_before_class_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = FinitePermutationAction(
        domain=tuple(f"p{i}" for i in range(5)),
        generators=(
            (1, 0, 2, 3, 4),
            (1, 2, 3, 4, 0),
        ),
    )

    def classes_must_not_be_computed(_: object) -> object:
        pytest.fail("over-order group reached class enumeration")

    monkeypatch.setattr(
        "jacobian.math.groups.characters.permutation.operations._classes_from_elements",
        classes_must_not_be_computed,
    )
    with pytest.raises(OperationResourceAdmissionError, match="order at most 64"):
        permutation_character(PermutationCharacterRequest(action=action))


def test_order_sixty_four_dihedral_action_fits_the_complete_boundary() -> None:
    degree = 32
    rotation = tuple((point + 1) % degree for point in range(degree))
    reflection = tuple((-point) % degree for point in range(degree))
    action = FinitePermutationAction(
        domain=tuple(f"v{point}" for point in range(degree)),
        generators=(rotation, reflection),
    )
    result = permutation_character(PermutationCharacterRequest(action=action))
    assert sum(result.axis.class_sizes) == 64
    assert len(result.values) <= 64
    assert sum(result.axis.class_sizes) == result.axis.group_order


def test_public_operation_is_discovered_and_class_function_compatible() -> None:
    assert len(TOOLS) == 1
    tool = TOOLS[0]
    assert tool.operation_id == "group.permutation_character.compute"
    request = PermutationCharacterRequest.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert isinstance(result, FiniteCharacter)
    assert result.values[0].coefficients[0].num == 3
    pairing = class_function_inner_product(result, result)
    assert pairing.inner_product.coefficients[0].as_fraction() == 2
