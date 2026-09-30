"""Exact checks for the degree of finite-group characters."""

from __future__ import annotations

import pytest

import jacobian.math.groups.characters.degree.operations as degree_operations
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters._models import (
    CharacterRingElement,
    CharacterRow,
    CharacterTableResult,
)
from jacobian.math.groups.characters.degree import (
    CharacterDegree,
)
from jacobian.math.groups.characters.operations import character_table
from jacobian.math.groups.operations import group_conjugacy_classes


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


def test_unsupported_source_order_rejects_before_class_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _s3_table()
    cycle = (*range(1, 61), 0)
    unsupported = PermutationGroup(degree=61, generators=(cycle,))
    extended_classes = tuple(
        tuple(element + tuple(range(3, 61)) for element in cls)
        for cls in table.partition.classes
    )
    partition = table.partition.model_copy(
        update={"source": unsupported, "classes": extended_classes}
    )
    axis = table.axis.model_copy(
        update={
            "group": unsupported,
            "class_representatives": tuple(cls[0] for cls in extended_classes),
        }
    )
    forged_table = table.model_copy(update={"partition": partition, "axis": axis})
    character = CharacterRingElement.model_construct(
        table=forged_table,
        irreducible_multiplicities=(1, 0, 0),
    )

    def fail_if_expanded(*_args: object, **_kwargs: object) -> None:
        pytest.fail("unsupported source order must reject before class expansion")

    monkeypatch.setattr(degree_operations, "group_conjugacy_classes", fail_if_expanded)
    with pytest.raises(OperationResourceAdmissionError):
        degree_operations.character_degree(character)


@pytest.mark.parametrize("missing", ["label", "degree"])
def test_missing_constructed_row_metadata_has_a_domain_error(missing: str) -> None:
    table = _s3_table()
    fields = dict(table.rows[0])
    del fields[missing]
    row = CharacterRow.model_construct(**fields)
    forged_table = table.model_copy(update={"rows": (row, *table.rows[1:])})
    character = CharacterRingElement.model_construct(
        table=forged_table, irreducible_multiplicities=(1, 0, 0)
    )

    with pytest.raises(OperationDomainValidationError) as error:
        degree_operations.character_degree(character)
    assert error.value.errors()[0]["loc"] == (
        "character",
        "table",
        "rows",
        "0",
        missing,
    )


@pytest.mark.parametrize(
    "owner,missing",
    [
        ("table", "degree_square_sum"),
        ("axis", "group_order"),
        ("axis", "cyclotomic_order"),
        ("group", "degree"),
        ("group", "generators"),
        ("coefficient", "num"),
        ("coefficient", "den"),
    ],
)
def test_missing_constructed_table_metadata_has_a_domain_error(
    owner: str, missing: str
) -> None:
    table = _s3_table().model_copy(deep=True)
    target = {
        "table": table,
        "axis": table.axis,
        "group": table.axis.group,
        "coefficient": table.rows[0].values[0].coefficients[0],
    }[owner]
    # The axis group may alias the partition source; construct a separate
    # claim so this exercises the retained metadata boundary specifically.
    if owner == "group":
        assert table.axis.group is not None
        target = table.axis.group.model_copy()
        table = table.model_copy(
            update={"axis": table.axis.model_copy(update={"group": target})}
        )
    assert target is not None
    del target.__dict__[missing]
    character = CharacterRingElement.model_construct(
        table=table, irreducible_multiplicities=(1, 0, 0)
    )

    with pytest.raises(OperationDomainValidationError):
        degree_operations.character_degree(character)


@pytest.mark.parametrize(
    "multiplicities", [(0, 0, 0), (1, 1, 2), (65, 0, 0), (10**512 - 1,) * 3]
)
def test_native_degree_round_trip_at_zero_and_coordinate_digit_boundary(
    multiplicities: tuple[int, ...],
) -> None:
    character = CharacterRingElement(
        table=_s3_table(), irreducible_multiplicities=multiplicities
    )
    result = degree_operations.character_degree(character)
    assert (
        result.degree == multiplicities[0] + multiplicities[1] + 2 * multiplicities[2]
    )
    decoded = CharacterDegree.model_validate_json(result.model_dump_json())
    assert degree_operations.character_degree(decoded.character) == decoded


def test_character_degree_expands_conjugacy_classes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.groups.operations as group_operations

    character = CharacterRingElement(
        table=_s3_table(), irreducible_multiplicities=(2, 1, 3)
    )
    calls = 0

    def count_expansion(
        degree: int, generators: list[list[int]]
    ) -> list[list[list[int]]]:
        nonlocal calls
        calls += 1
        return group_conjugacy_classes(degree, generators)

    monkeypatch.setattr(group_operations, "group_conjugacy_classes", count_expansion)
    monkeypatch.setattr(degree_operations, "group_conjugacy_classes", count_expansion)
    assert degree_operations.character_degree(character).degree == 9
    assert calls == 1


def test_structurally_valid_forged_row_degrees_are_not_authenticated() -> None:
    table = _s3_table()
    # Swap degrees while preserving the sum of squares. Shape validators
    # accept this claim, but the standard row still has identity value 2.
    forged_table = table.model_copy(
        update={
            "rows": (
                table.rows[0].model_copy(update={"degree": 2}),
                table.rows[1],
                table.rows[2].model_copy(update={"degree": 1}),
            )
        }
    )
    character = CharacterRingElement.model_validate_json(
        CharacterRingElement(
            table=forged_table, irreducible_multiplicities=(0, 0, 1)
        ).model_dump_json()
    )
    with pytest.raises(OperationDomainValidationError) as error:
        degree_operations.character_degree(character)
    assert error.value.errors()[0]["type"] == (
        "groups.characters.degree_noncanonical_table"
    )


def test_oversized_constructed_rows_reject_before_serialization_or_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _s3_table()
    table = table.model_copy(update={"rows": (table.rows[0],) * 10_000})
    character = CharacterRingElement.model_construct(
        table=table, irreducible_multiplicities=(1, 0, 0)
    )

    def fail_if_expanded(*_args: object, **_kwargs: object) -> None:
        pytest.fail("oversized raw shapes must reject before dumping or expansion")

    monkeypatch.setattr(CharacterRingElement, "model_dump", fail_if_expanded)
    monkeypatch.setattr(degree_operations, "group_conjugacy_classes", fail_if_expanded)
    with pytest.raises(OperationDomainValidationError) as error:
        degree_operations.character_degree(character)
    assert error.value.errors()[0]["type"] == "groups.characters.degree_table_shape"
