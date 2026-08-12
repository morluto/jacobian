from __future__ import annotations

from tests.component.checkers.exact_domain_checker_support import _request

from jacobian_checkers.finite_field_rank import (
    check_finite_field_linear_map_rank,
    check_finite_field_restriction,
)


def _rank_request(
    *,
    rank: int = 1,
    candidate_entries: list[list[int]] | None = None,
    candidate_direction_coordinates: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    presentation = {
        "characteristic": 2,
        "modulus_coefficients": [1, 1, 1],
        "generator": "a",
        "element_encoding_version": "power-basis-v1",
    }
    direction = {
        "presentation": presentation,
        "axis": {"name": "b", "labels": ["b1"]},
        "coordinates": [
            {"presentation": presentation, "coordinates": [1, 0]},
        ],
    }
    linear_map: dict[str, object] = {
        "source_axis": {"name": "source", "labels": ["B1"]},
        "target_axis": {"name": "target", "labels": ["y1", "y2"]},
        "matrix": {"prime": 2, "entries": [[1], [0]], "columns": 1},
    }
    matrix = linear_map["matrix"]
    assert isinstance(matrix, dict)
    candidate_map = dict(linear_map)
    candidate_map["matrix"] = {
        **matrix,
        "entries": candidate_entries or [[1], [0]],
    }
    candidate_direction = dict(direction)
    if candidate_direction_coordinates is not None:
        candidate_direction["coordinates"] = candidate_direction_coordinates
    return _request(
        "finite_field.linear_map.rank.compute",
        "finite-field.linear-map-rank.sympy-replay",
        {"direction": direction, "linear_map": linear_map},
        {
            "direction": candidate_direction,
            "linear_map": candidate_map,
            "rank": rank,
        },
    )


def test_sympy_checker_accepts_exact_prime_field_rank() -> None:
    decision = check_finite_field_linear_map_rank(_rank_request())

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


def test_sympy_checker_rejects_wrong_rank() -> None:
    decision = check_finite_field_linear_map_rank(_rank_request(rank=0))

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_sympy_checker_rejects_a_candidate_bound_to_another_map() -> None:
    decision = check_finite_field_linear_map_rank(
        _rank_request(candidate_entries=[[0], [0]])
    )

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_sympy_checker_rejects_a_candidate_bound_to_another_direction() -> None:
    decision = check_finite_field_linear_map_rank(
        _rank_request(
            candidate_direction_coordinates=[
                {
                    "presentation": {
                        "characteristic": 2,
                        "modulus_coefficients": [1, 1, 1],
                        "generator": "a",
                        "element_encoding_version": "power-basis-v1",
                    },
                    "coordinates": [0, 1],
                }
            ]
        )
    )

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def _restriction_request(*, output: list[list[int]]) -> dict[str, object]:
    presentation = {
        "characteristic": 2,
        "modulus_coefficients": [1, 1, 1],
        "generator": "a",
        "element_encoding_version": "power-basis-v1",
    }
    row_axis = {"name": "b", "labels": ["b1"]}
    column_axis = {"name": "y", "labels": ["y1"]}
    basis_axis = {"name": "basis", "labels": ["B1"]}
    element_a = {"presentation": presentation, "coordinates": [0, 1]}
    element_one = {"presentation": presentation, "coordinates": [1, 0]}
    claim = {
        "subspace": {
            "presentation": presentation,
            "basis_axis": basis_axis,
            "basis": [
                {
                    "presentation": presentation,
                    "row_axis": row_axis,
                    "column_axis": column_axis,
                    "entries": [[element_a]],
                }
            ],
        },
        "direction": {
            "presentation": presentation,
            "axis": row_axis,
            "coordinates": [element_one],
        },
    }
    candidate = {
        "source_axis": basis_axis,
        "target_axis": {"name": "Res(y)", "labels": ["y1:1", "y1:a"]},
        "matrix": {"prime": 2, "entries": output, "columns": 1},
    }
    return _request(
        "finite_field.restrict_scalars.compute",
        "finite-field.restriction.sympy-replay",
        claim,
        candidate,
    )


def test_sympy_checker_replays_restriction_of_scalars() -> None:
    decision = check_finite_field_restriction(_restriction_request(output=[[0], [1]]))

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


def test_sympy_checker_rejects_forged_restricted_map() -> None:
    decision = check_finite_field_restriction(_restriction_request(output=[[1], [0]]))

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
