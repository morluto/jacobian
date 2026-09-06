"""Defining-invariant tests for finite-field Paley tournaments."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields import (
    FiniteFieldPresentation,
    PaleyTournamentResult,
    _flint,
    finite_field,
    paley_tournament,
    verify_paley_tournament,
)

pytestmark = pytest.mark.requires_backend("flint")


def _encoded_coordinates(value: object, presentation: FiniteFieldPresentation) -> int:
    coordinates = _flint.coordinates(value, degree=presentation.degree)
    return sum(
        coordinate * presentation.characteristic**power
        for power, coordinate in enumerate(coordinates)
    )


def _assert_square_difference_invariant(result: PaleyTournamentResult) -> None:
    presentation = result.presentation
    context = _flint.context(presentation)
    elements = tuple(
        context(
            [
                (encoded // presentation.characteristic**power)
                % presentation.characteristic
                for power in range(presentation.degree)
            ]
        )
        for encoded in range(presentation.order)
    )
    squares = {
        _encoded_coordinates(value.square(), presentation) for value in elements[1:]
    }
    arcs = set(result.graph.edges)
    assert len(arcs) == presentation.order * (presentation.order - 1) // 2
    for left in range(presentation.order):
        for right in range(left + 1, presentation.order):
            assert ((left, right) in arcs) != ((right, left) in arcs)
            difference = _encoded_coordinates(
                elements[right] - elements[left], presentation
            )
            assert ((left, right) in arcs) == (difference in squares)


def test_f3_is_the_directed_three_cycle() -> None:
    result = paley_tournament(finite_field(3, (0, 1)))

    assert result.graph.vertex_count == 3
    assert tuple(value.coordinates for value in result.vertex_axis) == (
        (0,),
        (1,),
        (2,),
    )
    assert result.graph.edges == ((0, 1), (1, 2), (2, 0))
    assert result.orientation == "ARC_X_TO_Y_IFF_Y_MINUS_X_IS_NONZERO_SQUARE"
    assert (
        PaleyTournamentResult.model_validate(result.model_dump(mode="json")) == result
    )
    assert verify_paley_tournament(result)


def test_f7_has_the_complete_quadratic_residue_orientation() -> None:
    result = paley_tournament(finite_field(7, (0, 1)))

    _assert_square_difference_invariant(result)
    arcs = set(result.graph.edges)
    assert all(
        ((left + shift) % 7, (right + shift) % 7) in arcs
        for left, right in arcs
        for shift in range(7)
    )


def test_f27_uses_the_extension_field_difference() -> None:
    presentation = finite_field(3, (1, 2, 0, 1))

    first = paley_tournament(presentation)
    second = paley_tournament(presentation)

    _assert_square_difference_invariant(first)
    assert first.model_dump_json() == second.model_dump_json()


@pytest.mark.parametrize(
    "presentation",
    [
        finite_field(2, (0, 1)),
        finite_field(5, (0, 1)),
    ],
)
def test_rejects_orders_outside_three_modulo_four(
    presentation: FiniteFieldPresentation,
) -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        paley_tournament(presentation)
    assert (
        error.value.errors()[0]["type"]
        == "finite_field.paley_tournament_order_congruent_to_three_mod_four"
    )


def test_rejects_a_complete_tournament_beyond_the_edge_envelope() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        paley_tournament(finite_field(2039, (0, 1)))
    assert (
        error.value.errors()[0]["type"]
        == "finite_field.paley_tournament_exceeds_graph_edge_envelope"
    )


def test_rejects_a_tournament_beyond_the_directed_graph_edge_envelope() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        paley_tournament(finite_field(587, (0, 1)))
    assert (
        error.value.errors()[0]["type"]
        == "finite_field.paley_tournament_exceeds_graph_edge_envelope"
    )


def test_presentation_metadata_is_not_charged_as_mathematical_output() -> None:
    presentation = finite_field(3, (0, 1), generator="a" * 10_485_600)

    result = paley_tournament(presentation)

    assert result.presentation.generator == presentation.generator


def test_serialized_result_is_structural_and_publicly_verifiable() -> None:
    result = paley_tournament(finite_field(3, (0, 1)))
    payload = result.model_dump(mode="json")
    payload["graph"]["vertex_count"] = 4
    decoded = PaleyTournamentResult.model_validate(payload)
    assert not verify_paley_tournament(decoded)

    payload = result.model_dump(mode="json")
    payload["graph"]["edges"] = [[0, 1], [1, 0], [2, 0]]
    decoded = PaleyTournamentResult.model_validate(payload)
    assert not verify_paley_tournament(decoded)

    payload = result.model_dump(mode="json")
    payload["graph"]["edges"] = list(reversed(payload["graph"]["edges"]))
    decoded = PaleyTournamentResult.model_validate(payload)
    assert not verify_paley_tournament(decoded)

    payload = result.model_dump(mode="json")
    payload["vertex_axis"][0]["coordinates"] = [1]
    decoded = PaleyTournamentResult.model_validate(payload)
    assert not verify_paley_tournament(decoded)
