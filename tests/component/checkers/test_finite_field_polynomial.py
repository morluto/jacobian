from __future__ import annotations

from copy import deepcopy
from typing import Any

from tests.component.checkers.exact_domain_checker_support import _request

from jacobian_checkers.finite_field_polynomial import (
    check_finite_map_collision,
    check_finite_map_fibers,
    check_finite_map_permutation,
    check_finite_map_table,
)

_PRESENTATION = {
    "characteristic": 2,
    "modulus_coefficients": [1, 1, 1],
    "generator": "a",
    "element_encoding_version": "power-basis-v1",
}


def _element(coordinates: list[int]) -> dict[str, Any]:
    return {"presentation": _PRESENTATION, "coordinates": coordinates}


def _polynomial_map(exponent: int) -> dict[str, Any]:
    zero = _element([0, 0])
    one = _element([1, 0])
    return {
        "domain": _PRESENTATION,
        "codomain": _PRESENTATION,
        "polynomial": {
            "presentation": _PRESENTATION,
            "variable": "x",
            "coefficients": [
                one if power == exponent else zero for power in range(exponent + 1)
            ],
        },
    }


def _table(exponent: int) -> dict[str, Any]:
    outputs = (
        [[0, 0], [1, 0], [1, 0], [1, 0]]
        if exponent == 3
        else [[0, 0], [1, 0], [1, 1], [0, 1]]
    )
    return {
        "map": _polynomial_map(exponent),
        "entries": [
            [_element([(encoded // 1) % 2, (encoded // 2) % 2]), _element(output)]
            for encoded, output in enumerate(outputs)
        ],
    }


def _bound(
    operation_id: str,
    witness_format: str,
    claim: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, object]:
    return _request(operation_id, witness_format, claim, candidate)


def test_sympy_checker_replays_complete_finite_map_table() -> None:
    table = _table(3)
    decision = check_finite_map_table(
        _bound(
            "finite_field.polynomial_map.table.compute",
            "finite-field.polynomial-map-table.sympy-replay",
            {"polynomial_map": table["map"]},
            table,
        )
    )

    assert decision["accepted"] is True
    assert decision["coverage"] == "EXHAUSTIVE"


def test_sympy_checker_rejects_forged_finite_map_table() -> None:
    table = _table(3)
    forged = deepcopy(table)
    forged["entries"][2][1] = _element([0, 0])

    decision = check_finite_map_table(
        _bound(
            "finite_field.polynomial_map.table.compute",
            "finite-field.polynomial-map-table.sympy-replay",
            {"polynomial_map": table["map"]},
            forged,
        )
    )

    assert decision["accepted"] is False


def test_sympy_checker_replays_fibers_and_collision() -> None:
    table = _table(3)
    zero, one, a, one_plus_a = [source for source, _ in table["entries"]]
    fibers = {
        "table": table,
        "fibers": [[zero, [zero]], [one, [one, a, one_plus_a]]],
    }
    collision = {
        "table": table,
        "left": one,
        "right": a,
        "image": one,
    }

    fiber_decision = check_finite_map_fibers(
        _bound(
            "finite_field.polynomial_map.fibers.compute",
            "finite-field.polynomial-map-fibers.sympy-replay",
            {"table": table},
            fibers,
        )
    )
    collision_decision = check_finite_map_collision(
        _bound(
            "finite_field.polynomial_map.collision.compute",
            "finite-field.polynomial-map-collision.sympy-replay",
            {"table": table},
            collision,
        )
    )

    assert fiber_decision["accepted"] is True
    assert collision_decision["accepted"] is True


def test_sympy_checker_replays_permutation_inverse() -> None:
    table = _table(2)
    inverse = sorted(
        table["entries"],
        key=lambda entry: entry[1]["coordinates"][0] + 2 * entry[1]["coordinates"][1],
    )
    candidate = {
        "table": table,
        "inverse_entries": [[target, source] for source, target in inverse],
    }

    decision = check_finite_map_permutation(
        _bound(
            "finite_field.polynomial_map.permutation.compute",
            "finite-field.polynomial-map-permutation.sympy-replay",
            {"table": table},
            candidate,
        )
    )

    assert decision["accepted"] is True
