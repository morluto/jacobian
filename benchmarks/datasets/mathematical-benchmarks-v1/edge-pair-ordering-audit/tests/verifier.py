import itertools
import json
import math
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)


def exhaustive(n):
    edges = list(itertools.combinations(range(n), 2))
    total = 0
    for mask in range(1 << len(edges)):
        chosen = [edges[i] for i in range(len(edges)) if mask >> i & 1]
        total += sum(
            len(set(first) & set(second)) == 1
            for first in chosen
            for second in chosen
            if first != second
        )
    return total


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, ValueError, RecursionError, MemoryError):
        return {}
    return value if isinstance(value, dict) else {}


def _exact_json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's bool/int equality coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return bool(
            set(left) == set(right)
            and all(_exact_json_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_json_equal(item, other)
            for item, other in zip(left, right, strict=True)
        )
    return left == right


def _formula_shape_is_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "incident_vertex_offsets",
        "free_edge_base",
        "free_edge_exponent",
    }:
        return False
    offsets = value["incident_vertex_offsets"]
    exponent = value["free_edge_exponent"]
    return bool(
        isinstance(offsets, list)
        and len(offsets) == 3
        and len(set(offsets)) == 3
        and all(type(offset) is int for offset in offsets)
        and type(value["free_edge_base"]) is int
        and isinstance(exponent, dict)
        and set(exponent) == {"binomial_order", "offset"}
        and type(exponent["binomial_order"]) is int
        and type(exponent["offset"]) is int
    )


def _result_shape_is_valid(result: object) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "pair_semantics",
        "formula",
        "probe_values",
    }:
        return False
    if not isinstance(result["pair_semantics"], str):
        return False
    if result["pair_semantics"] not in {"ORDERED", "UNORDERED"}:
        return False
    if not _formula_shape_is_valid(result["formula"]):
        return False
    probes = result["probe_values"]
    if not isinstance(probes, list) or len(probes) != 4:
        return False
    rows: list[tuple[int, int]] = []
    for probe in probes:
        if not isinstance(probe, dict) or set(probe) != {"n", "coefficient"}:
            return False
        n = probe["n"]
        coefficient = probe["coefficient"]
        if type(n) is not int or not 3 <= n <= 6:
            return False
        if type(coefficient) is not int or coefficient < 1:
            return False
        rows.append((n, coefficient))
    return len({n for n, _ in rows}) == len(rows) == 4 and {n for n, _ in rows} == {
        3,
        4,
        5,
        6,
    }


def _mathematical_result_is_valid(result: object, source: dict[str, Any]) -> bool:
    expected_source = {
        "task_id": "jacobian/edge-pair-ordering-audit",
        "pair_semantics": "ordered",
        "edge_pair_condition": "e1 != e2 and |e1 intersection e2| = 1",
        "graph_family": "all labeled simple graphs on n vertices",
        "polynomial_definition": (
            "For each labeled simple graph G and ordered pair (e1,e2) of "
            "distinct edges, p_(G,e1,e2)(x) = x when |e1 intersection e2| = 1 "
            "and p_(G,e1,e2)(x) = 0 otherwise; sum these polynomials over all "
            "G and all ordered edge pairs."
        ),
        "source_derivation": (
            "The original source treats the double sum as unordered, counting "
            "each incident edge pair once rather than in both orientations, "
            "yielding half the correct coefficient."
        ),
        "probe_n": [3, 4, 5, 6],
    }
    if not _exact_json_equal(source, expected_source):
        return False
    if not isinstance(result, dict):
        return False
    try:
        probes = result["probe_values"]
        if not isinstance(probes, list) or len(probes) > 10:
            return False
        values: dict[int, int] = {}
        for probe in probes:
            if not isinstance(probe, dict):
                return False
            n = probe.get("n")
            coefficient = probe.get("coefficient")
            if type(n) is not int or type(coefficient) is not int:
                return False
            values[n] = coefficient
        formula = result["formula"]
        offsets = formula["incident_vertex_offsets"]
        base = formula["free_edge_base"]
        exponent = formula["free_edge_exponent"]

        def submitted_formula(n: int) -> int:
            incident_factor = math.prod(n + offset for offset in offsets)
            free_edge_exponent = (
                math.comb(n, exponent["binomial_order"]) + exponent["offset"]
            )
            return incident_factor * base**free_edge_exponent

        return bool(
            result["pair_semantics"] == "ORDERED"
            and sorted(offsets) == [-2, -1, 0]
            and base == 2
            and exponent == {"binomial_order": 2, "offset": -2}
            and set(values) == {3, 4, 5, 6}
            and all(values[n] == exhaustive(n) == submitted_formula(n) for n in values)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _protocol_is_valid(submission: object, contract: bool, result: object) -> bool:
    if not isinstance(submission, dict):
        return False
    return _result_shape_is_valid(result)


def main():
    _input_binding = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    data = submission if isinstance(submission, dict) else {}
    source = _load_json(Path("/tests/input.json"))
    result = data.get("result")
    protocol = _protocol_is_valid(submission, True, result)
    math_correct = _mathematical_result_is_valid(result, source)
    reward = 1.0 if _input_binding and protocol and math_correct else 0.0
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(protocol),
                "correctness": float(math_correct),
                "input_binding": float(_input_binding),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
