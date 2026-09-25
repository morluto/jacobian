from __future__ import annotations

from itertools import combinations, product

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology._models import SimplicialComplexRequest
from jacobian.math.topology.discrete_morse import (
    MinimumMorseMatchingRequest,
    MinimumMorseMatchingResult,
    minimum_matching,
)
from jacobian.math.topology.discrete_morse._tools import TOOLS

type Cell = tuple[str, ...]
type Cover = tuple[Cell, Cell]

TRIANGLE = {
    "vertices": ["a", "b", "c"],
    "facets": [["a", "b", "c"]],
}


def _request(complex_data: dict[str, object]) -> MinimumMorseMatchingRequest:
    return MinimumMorseMatchingRequest(
        complex=SimplicialComplexRequest.model_validate(complex_data)
    )


def _is_acyclic(
    cells: tuple[Cell, ...], covers: tuple[Cover, ...], chosen: tuple[Cover, ...]
) -> bool:
    matched = set(chosen)
    adjacency: dict[Cell, list[Cell]] = {cell: [] for cell in cells}
    for face, coface in covers:
        if (face, coface) in matched:
            adjacency[face].append(coface)
        else:
            adjacency[coface].append(face)
    color: dict[Cell, int] = {}

    def visit(cell: Cell) -> bool:
        color[cell] = 1
        for next_cell in adjacency[cell]:
            if color.get(next_cell) == 1:
                return False
            if color.get(next_cell, 0) == 0 and not visit(next_cell):
                return False
        color[cell] = 2
        return True

    return all(color.get(cell, 0) != 0 or visit(cell) for cell in cells)


def _brute_force_minimum(facets: tuple[Cell, ...]) -> int:
    cells: set[Cell] = {(vertex,) for facet in facets for vertex in facet}
    cells.update(
        tuple(sorted(face))
        for facet in facets
        for size in range(2, len(facet) + 1)
        for face in combinations(facet, size)
    )
    ordered = tuple(sorted(cells, key=lambda cell: (len(cell), cell)))
    covers = tuple(
        (face, coface)
        for coface in ordered
        if len(coface) > 1
        for face in combinations(coface, len(coface) - 1)
    )
    best = len(ordered)
    for selected_flags in product((False, True), repeat=len(covers)):
        chosen = tuple(
            edge
            for edge, selected in zip(covers, selected_flags, strict=True)
            if selected
        )
        used = {cell for edge in chosen for cell in edge}
        if len(used) != 2 * len(chosen):
            continue
        if _is_acyclic(ordered, covers, chosen):
            best = min(best, len(ordered) - 2 * len(chosen))
    return best


def test_minimum_matching_matches_independent_exhaustive_oracle() -> None:
    declaration = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "topology.discrete_morse.matching.minimum.compute"
    )
    result = declaration.run(_request(TRIANGLE))

    assert isinstance(result, MinimumMorseMatchingResult)
    assert result.minimum_critical_cell_count == 1
    assert result.minimum_critical_cell_count == _brute_force_minimum(
        (("a", "b", "c"),)
    )
    assert result.matching.critical_profile is not None
    assert len(result.matching.critical_profile.critical_cells) == 1
    assert result.matching.outcome == "ACYCLIC_MATCHING"
    assert (
        MinimumMorseMatchingResult.model_validate(result.model_dump(mode="json"))
        == result
    )


def test_minimum_matching_is_published_and_empty_search_is_exact() -> None:
    declaration = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "topology.discrete_morse.matching.minimum.compute"
    )
    assert declaration.request_type is MinimumMorseMatchingRequest
    assert declaration.result_type is MinimumMorseMatchingResult
    point = minimum_matching(_request({"vertices": ["a"], "facets": [["a"]]}))
    assert point.minimum_critical_cell_count == 1
    assert point.matching.pairs == ()


def test_minimum_matching_rejects_search_before_expansion() -> None:
    request = _request(
        {
            "vertices": ["a", "b", "c", "d"],
            "facets": [["a", "b", "c", "d"]],
        }
    )

    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        minimum_matching(request)

    assert (
        exc_info.value.errors()[0]["type"]
        == "topology.discrete_morse.minimum_matching.admission.search_work"
    )
