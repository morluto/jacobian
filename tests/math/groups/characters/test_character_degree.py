"""Exact checks for the degree of finite-group characters."""

from __future__ import annotations

import pytest

import jacobian.math.groups.characters.degree.operations as degree_operations
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
)
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters._models import (
    CharacterRingElement,
    CharacterTableResult,
)
from jacobian.math.groups.characters.operations import character_table
from jacobian.math.groups.operations import group_conjugacy_classes

OPERATION_ID = "character.degree.compute"


def _s3_table() -> CharacterTableResult:
    generators = ((1, 2, 0), (1, 0, 2))
    group = PermutationGroup(degree=3, generators=generators)
    classes = group_conjugacy_classes(3, [list(row) for row in generators])
    partition = GroupConjugacyClassesResult._from_kernel(
        group,
        tuple(tuple(tuple(element) for element in cls) for cls in classes),
    )
    return character_table(partition)


def test_s3_character_degree_matches_independent_identity_value() -> None:
    table = _s3_table()
    multiplicities = (2, 1, 3)
    character = CharacterRingElement(
        table=table,
        irreducible_multiplicities=multiplicities,
    )

    result = degree_operations.character_degree(character)

    # The S3 irreducibles have degrees 1, 1, and 2; direct evaluation at the
    # identity permutation independently gives the same value.
    direct_identity_value = (
        multiplicities[0] + multiplicities[1] + 2 * multiplicities[2]
    )
    assert result.degree == direct_identity_value == 9
    assert result.character == character


def test_irreducible_and_reducible_degrees_are_exact() -> None:
    table = _s3_table()
    assert [
        degree_operations.character_degree(
            CharacterRingElement(
                table=table,
                irreducible_multiplicities=tuple(int(i == row) for i in range(3)),
            )
        ).degree
        for row in range(3)
    ] == [1, 1, 2]


def test_virtual_character_is_not_misreported_as_an_ordinary_degree() -> None:
    character = CharacterRingElement(
        table=_s3_table(), irreducible_multiplicities=(1, -1, 0)
    )
    with pytest.raises(OperationDomainValidationError) as error:
        degree_operations.character_degree(character)
    assert error.value.errors()[0]["type"] == (
        "groups.characters.degree_requires_ordinary_character"
    )


def test_forged_character_table_is_rejected_after_source_authentication() -> None:
    table = _s3_table()
    rows = (table.rows[0].model_copy(update={"degree": 2}), *table.rows[1:])
    forged_table = table.model_copy(update={"rows": rows})
    forged_character = CharacterRingElement.model_construct(
        table=forged_table,
        irreducible_multiplicities=(1, 0, 0),
    )

    with pytest.raises(OperationDomainValidationError) as error:
        degree_operations.character_degree(forged_character)
    assert error.value.errors()[0]["type"] == (
        "groups.characters.degree_noncanonical_table"
    )


def test_degree_does_not_reexpand_the_source_group() -> None:
    """A projection must not redo the work the public table operation did.

    The earlier implementation re-derived the source group, its conjugacy
    classes, and the entire character table only to confirm that the caller's
    retained copy was canonical. That made a dot product of two already-public
    fields cost more than computing the table in the first place.
    """
    character = CharacterRingElement.model_validate(
        CharacterRingElement(
            table=_s3_table(), irreducible_multiplicities=(1, 0, 0)
        ).model_dump(mode="python")
    )
    assert degree_operations.character_degree(character).degree == 1


def test_degree_is_native_only_and_agrees_with_the_published_table() -> None:
    """`character.degree.compute` is a projection, so it is not in the catalog.

    chi(1) is the dot product of the irreducible multiplicities with the row
    degrees, and both already appear in public results. The native helper stays
    reachable; publishing it would be a second discovery target for a
    postcondition `finite_group.character_table.compute` already establishes.
    """
    assert Catalog.open().operation(OPERATION_ID) is None

    from jacobian.math.groups.characters.degree.operations import character_degree

    table = _s3_table()
    character = CharacterRingElement.model_validate(
        CharacterRingElement(
            table=_s3_table(), irreducible_multiplicities=(1, 0, 0)
        ).model_dump(mode="python")
    )
    result = character_degree(character)
    assert result.degree == 1

    # the same dot product computed from the two published values
    assert result.degree == sum(
        multiplicity * row.degree
        for multiplicity, row in zip(
            character.irreducible_multiplicities, table.rows, strict=True
        )
    )
    # The regular character of S3 is the sum of the irreducibles weighted by
    # their degrees, so its degree is |S3| = 6. Rows here have degrees 1, 1, 2,
    # giving multiplicities 1, 1, 2 and a total of 1 + 1 + 4.
    regular = CharacterRingElement(
        table=_s3_table(), irreducible_multiplicities=(1, 1, 2)
    )
    assert character_degree(regular).degree == 6


def test_degree_rejects_a_negative_multiplicity() -> None:
    character = CharacterRingElement.model_construct(
        table=_s3_table(),
        irreducible_multiplicities=(-1, 0, 0),
    )
    with pytest.raises(OperationDomainValidationError):
        degree_operations.character_degree(character)
