import itertools
import json
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
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


def _result_shape_is_valid(result: object) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "pair_semantics",
        "incident_ordered_pair_factor",
        "free_edge_factor_exponent",
        "formula",
        "probe_values",
    }:
        return False
    if not isinstance(result["pair_semantics"], str):
        return False
    if result["pair_semantics"] not in {"ORDERED", "UNORDERED"} or not all(
        type(result[field]) is str and bool(result[field].strip())
        for field in (
            "incident_ordered_pair_factor",
            "free_edge_factor_exponent",
            "formula",
        )
    ):
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
        return bool(
            result["pair_semantics"] == "ORDERED"
            and result["incident_ordered_pair_factor"] == "n(n-1)(n-2)"
            and result["free_edge_factor_exponent"] == "binom(n,2)-2"
            and result["formula"] == "n(n-1)(n-2)*2^(binom(n,2)-2)"
            and set(values) == {3, 4, 5, 6}
            and all(
                values[n]
                == exhaustive(n)
                == n * (n - 1) * (n - 2) * 2 ** (n * (n - 1) // 2 - 2)
                for n in values
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _protocol_is_valid(submission: object, contract: bool, result: object) -> bool:
    if not isinstance(submission, dict):
        return False
    return _result_shape_is_valid(result)


def main():
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    source = _load_json(Path("/tests/input.json"))
    result = data.get("result")
    protocol = _protocol_is_valid(submission, True, result)
    math_correct = _mathematical_result_is_valid(result, source)
    reward = 1.0 if protocol and math_correct else 0.0
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(protocol),
                "correctness": float(math_correct),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
