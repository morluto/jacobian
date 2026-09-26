"""Exact second lambda operations in bounded finite representation rings."""

from itertools import permutations

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters._models import (
    CharacterExteriorSquareRequest,
    CharacterRingElement,
    CharacterSymmetricSquareRequest,
)
from jacobian.math.groups.characters.operations import character_table
from jacobian.math.groups.characters.representation_ring_operations import (
    character_exterior_square,
    character_symmetric_square,
)
from jacobian.math.groups.operations import group_conjugacy_classes


def _s3_table():
    source = PermutationGroup(degree=3, generators=((1, 2, 0), (1, 0, 2)))
    classes = group_conjugacy_classes(3, [list(g) for g in source.generators])
    partition = GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(g) for g in cls) for cls in classes)
    )
    return character_table(partition)


def _element(table, coordinates):
    return CharacterRingElement(
        table=table, irreducible_multiplicities=tuple(coordinates)
    )


def _independent_s3_oracle() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Average independently defined module characters over the six elements."""
    symmetric = [0, 0, 0]
    exterior = [0, 0, 0]
    for permutation in permutations(range(3)):
        fixed = sum(permutation[i] == i for i in range(3))
        standard = fixed - 1  # permutation module modulo its invariant line
        squared = tuple(permutation[permutation[i]] for i in range(3))
        squared_fixed = sum(squared[i] == i for i in range(3))
        standard_square = squared_fixed - 1
        sign = (
            -1
            if sum(
                permutation[i] > permutation[j]
                for i in range(3)
                for j in range(i + 1, 3)
            )
            % 2
            else 1
        )
        irreducibles = (1, sign, standard)
        symmetric_value = (standard * standard + standard_square) // 2
        exterior_value = (standard * standard - standard_square) // 2
        for row, value in enumerate(irreducibles):
            symmetric[row] += symmetric_value * value
            exterior[row] += exterior_value * value
    return tuple(value // 6 for value in symmetric), tuple(
        value // 6 for value in exterior
    )


def test_s3_lambda_squares_match_direct_elementwise_oracle() -> None:
    table = _s3_table()
    standard = _element(table, (0, 0, 1))
    symmetric = character_symmetric_square(
        CharacterSymmetricSquareRequest(character=standard)
    )
    exterior = character_exterior_square(
        CharacterExteriorSquareRequest(character=standard)
    )
    oracle = _independent_s3_oracle()
    assert symmetric.irreducible_multiplicities == (1, 0, 1)
    assert exterior.irreducible_multiplicities == (0, 1, 0)
    assert symmetric.irreducible_multiplicities == oracle[0]
    assert exterior.irreducible_multiplicities == oracle[1]
    assert (
        CharacterRingElement.model_validate_json(symmetric.model_dump_json())
        == symmetric
    )
    assert (
        CharacterRingElement.model_validate_json(exterior.model_dump_json()) == exterior
    )


def test_cyclic_squares_and_signed_virtual_characters() -> None:
    degree = 5
    generator = tuple((point + 1) % degree for point in range(degree))
    classes = group_conjugacy_classes(degree, [list(generator)])
    partition = GroupConjugacyClassesResult._from_kernel(
        PermutationGroup(degree=degree, generators=(generator,)),
        tuple(tuple(tuple(g) for g in cls) for cls in classes),
    )
    table = character_table(partition)
    character = _element(table, (0, 1, 0, 0, 0))
    # For C5, Sym^2(rho_1)=rho_2 and Lambda^2(rho_1)=0.
    assert character_symmetric_square(
        CharacterSymmetricSquareRequest(character=character)
    ).irreducible_multiplicities == (0, 0, 1, 0, 0)
    assert character_exterior_square(
        CharacterExteriorSquareRequest(character=character)
    ).irreducible_multiplicities == (0, 0, 0, 0, 0)
    virtual = _element(table, (1, -1, 0, 0, 0))
    assert (
        len(
            character_symmetric_square(
                CharacterSymmetricSquareRequest(character=virtual)
            ).irreducible_multiplicities
        )
        == 5
    )
    assert (
        len(
            character_exterior_square(
                CharacterExteriorSquareRequest(character=virtual)
            ).irreducible_multiplicities
        )
        == 5
    )


@pytest.mark.parametrize(
    ("operation_id", "request_type"),
    [
        ("character.symmetric_square.compute", CharacterSymmetricSquareRequest),
        ("character.exterior_square.compute", CharacterExteriorSquareRequest),
    ],
)
def test_lambda_square_operations_are_published(operation_id, request_type) -> None:
    tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == operation_id)
    assert tool.request_type is request_type
    assert tool.result_type is CharacterRingElement
    example = tool.examples[0]
    result = invoke_operation(operation_id, example.input, Catalog.open())
    assert result.output["irreducible_multiplicities"] == (
        [1, 0, 1] if "symmetric" in operation_id else [0, 1, 0]
    )
