"""Exact Dynkin edge labels against the source Cartan matrix."""

from __future__ import annotations

import json

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.groups.root_systems import DynkinEdge, FiniteDynkinDiagram
from jacobian.math.groups.root_systems._models import CartanMatrix, CartanMatrixRequest
from jacobian.math.groups.root_systems._tools import TOOLS
from jacobian.math.groups.root_systems.operations import dynkin_diagram


def _cartan_from_labeled_graph(
    result: FiniteDynkinDiagram,
) -> tuple[tuple[int, ...], ...]:
    """Reconstruct the Cartan matrix only from graph nodes and ordered edges."""
    rank = len(result.simple_root_axis)
    entries = [[int(i == j) * 2 for j in range(rank)] for i in range(rank)]
    for edge in result.edges:
        left, right = edge.simple_root_indices
        entries[left][right], entries[right][left] = edge.cartan_pairing
    return tuple(tuple(row) for row in entries)


@pytest.mark.parametrize(
    ("matrix", "pair", "multiplicity"),
    [
        (((2, -2), (-1, 2)), (-2, -1), 2),
        (((2, -1), (-2, 2)), (-1, -2), 2),
        (((2, -3), (-1, 2)), (-3, -1), 3),
    ],
)
def test_multiple_edge_direction_and_multiplicity_are_exact(
    matrix: tuple[tuple[int, ...], ...],
    pair: tuple[int, int],
    multiplicity: int,
) -> None:
    result = dynkin_diagram(CartanMatrix.model_validate(matrix))
    assert result.simple_root_axis == (0, 1)
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge == DynkinEdge(
        simple_root_indices=(0, 1),
        cartan_pairing=pair,
        edge_multiplicity=multiplicity,
    )
    assert _cartan_from_labeled_graph(result) == matrix
    assert FiniteDynkinDiagram.model_validate_json(result.model_dump_json()) == result


def test_reducible_a1_times_a1_keeps_both_isolated_nodes() -> None:
    matrix = ((2, 0), (0, 2))
    result = dynkin_diagram(CartanMatrix.model_validate(matrix))
    assert result.simple_root_axis == (0, 1)
    assert result.edges == ()
    assert _cartan_from_labeled_graph(result) == matrix


def test_diagram_preflights_before_datum_and_edge_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.groups.root_systems.operations as operations

    monkeypatch.setattr(operations, "MAX_DYNKIN_DIAGRAM_WORK", 0)
    monkeypatch.setattr(
        operations,
        "_cartan_datum_from_admitted",
        lambda _cartan: pytest.fail("datum constructed before admission"),
    )
    with pytest.raises(OperationResourceAdmissionError):
        operations.dynkin_diagram(CartanMatrix.model_validate(((2, -1), (-1, 2))))


def test_manifest_example_invokes_the_dynkin_operation() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "root_system.dynkin_diagram.compute"
    )
    request = CartanMatrixRequest.model_validate_json(
        json.dumps(operation.examples[0].input), strict=True
    )
    result = operation.run(request)
    assert result.edges[0].cartan_pairing == (-3, -1)
    assert result.edges[0].edge_multiplicity == 3
