"""Exact highest-weight characters and independent finite checks."""

import json

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.root_systems import (
    HighestWeightCharacterRequest,
    IrreducibleWeightCharacter,
    WeightMultiplicity,
    highest_weight_character,
    weyl_dimension,
    weyl_weight_orbit,
)
from jacobian.math.groups.root_systems._tools import TOOLS

A1 = ((2,),)
A2 = ((2, -1), (-1, 2))
A3 = ((2, -1, 0), (-1, 2, -1), (0, -1, 2))
B2 = ((2, -2), (-1, 2))


def _terms(character: IrreducibleWeightCharacter) -> dict[tuple[int, ...], int]:
    return {term.weight: term.multiplicity for term in character.terms}


@pytest.mark.parametrize(
    ("cartan", "highest", "expected"),
    (
        (
            A1,
            (3,),
            {(-3,): 1, (-1,): 1, (1,): 1, (3,): 1},
        ),
        (
            A2,
            (1, 1),
            {
                (-2, 1): 1,
                (-1, -1): 1,
                (-1, 2): 1,
                (0, 0): 2,
                (1, -2): 1,
                (1, 1): 1,
                (2, -1): 1,
            },
        ),
        (
            A3,
            (1, 0, 1),
            {
                (-2, 1, 0): 1,
                (-1, -1, 1): 1,
                (-1, 0, -1): 1,
                (-1, 1, 1): 1,
                (-1, 2, -1): 1,
                (0, -1, 2): 1,
                (0, 0, 0): 3,
                (0, 1, -2): 1,
                (1, -2, 1): 1,
                (1, -1, -1): 1,
                (1, 0, 1): 1,
                (1, 1, -1): 1,
                (2, -1, 0): 1,
            },
        ),
    ),
)
def test_hand_tables(cartan, highest, expected) -> None:
    result = highest_weight_character(cartan, highest)
    assert _terms(result) == expected
    assert result.highest_weight == highest
    assert result.weight_axis == tuple(range(len(highest)))


@pytest.mark.parametrize(("cartan", "highest"), ((A2, (2, 1)), (A3, (1, 1, 0))))
def test_character_has_weyl_symmetry_and_dimension(cartan, highest) -> None:
    character = highest_weight_character(cartan, highest)
    multiplicities = _terms(character)
    for weight, multiplicity in multiplicities.items():
        orbit = weyl_weight_orbit(cartan, weight)
        assert all(multiplicities[other] == multiplicity for other in orbit.orbit)
    assert sum(multiplicities.values()) == weyl_dimension(cartan, highest).dimension


def test_character_is_a_canonical_round_tripping_value() -> None:
    value = highest_weight_character(A2, (1, 1))
    revived = IrreducibleWeightCharacter.model_validate_json(value.model_dump_json())
    assert revived == value
    assert isinstance(value.terms[0], WeightMultiplicity)


def test_character_value_retains_its_supported_type() -> None:
    valid = highest_weight_character(A2, (1, 1))
    payload = valid.model_dump()
    payload["matrix"] = type(valid.matrix).model_validate(B2)
    with pytest.raises(ValueError, match="character terms"):
        IrreducibleWeightCharacter.model_validate(payload)


def test_request_rejects_wrong_rank_and_negative_highest_weight() -> None:
    with pytest.raises(ValueError, match="highest weight"):
        HighestWeightCharacterRequest(matrix=A2, highest_weight=(1,))
    with pytest.raises(ValueError, match="highest weight"):
        HighestWeightCharacterRequest(matrix=A1, highest_weight=(-1,))


def test_non_type_a_finite_data_is_rejected_without_a_partial_table() -> None:
    with pytest.raises(
        OperationDomainValidationError,
        match="irreducible type A only",
    ):
        highest_weight_character(B2, (1, 0))


def test_candidate_state_bound_precedes_enumeration() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="candidate-state"):
        highest_weight_character(A1, (MAX_STATE_BOUND,))


def test_work_admission_precedes_candidate_expansion(monkeypatch) -> None:
    import jacobian.math.groups.root_systems.weight_character as operation

    monkeypatch.setattr(operation, "MAX_CHARACTER_WORK", 0)
    monkeypatch.setattr(
        operation,
        "_type_a_candidates",
        lambda *_args: pytest.fail("candidate weights expanded before admission"),
    )
    with pytest.raises(OperationResourceAdmissionError, match="work"):
        highest_weight_character(A2, (1, 0))


MAX_STATE_BOUND = 4_096


def test_manifest_example_executes_the_public_operation() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "root_system.highest_weight_character.compute"
    )
    request = HighestWeightCharacterRequest.model_validate_json(
        json.dumps(tool.examples[0].input), strict=True
    )
    result = tool.run(request)
    assert sum(term.multiplicity for term in result.terms) == 8


def _small_semistandard_tableau_character(
    highest: tuple[int, ...],
) -> dict[tuple[int, ...], int]:
    """Brute-force small tableaux independently as a Kostka-number oracle."""
    n = len(highest) + 1
    shape = (*(sum(highest[index:]) for index in range(len(highest))), 0)
    cells = tuple(
        (row, column) for row, width in enumerate(shape) for column in range(width)
    )
    contents: dict[tuple[int, ...], int] = {}
    filling: dict[tuple[int, int], int] = {}

    def visit(position: int) -> None:
        if position == len(cells):
            content = tuple(
                sum(value == letter for value in filling.values())
                for letter in range(n)
            )
            weight = tuple(
                content[index] - content[index + 1] for index in range(n - 1)
            )
            contents[weight] = contents.get(weight, 0) + 1
            return
        row, column = cells[position]
        lower = 0
        if column:
            lower = filling[row, column - 1]
        if row and column < shape[row - 1]:
            lower = max(lower, filling[row - 1, column] + 1)
        for value in range(lower, n):
            filling[row, column] = value
            visit(position + 1)
        filling.pop((row, column), None)

    visit(0)
    return contents


@pytest.mark.parametrize(
    ("cartan", "highest"),
    ((A1, (2,)), (A2, (1, 1)), (A2, (2, 1)), (A3, (1, 0, 1))),
)
def test_freudenthal_table_matches_independent_semistandard_tableaux(
    cartan, highest
) -> None:
    assert _terms(highest_weight_character(cartan, highest)) == (
        _small_semistandard_tableau_character(highest)
    )
