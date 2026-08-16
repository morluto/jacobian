from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from verifier_support import (
    json_value_equal,
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/thrackle-local-maximality-certificate"
POINTS = [(0, 0), (4, 0), (5, 3), (2, 5), (-1, 3)]
ALL = list(combinations(range(5), 2))


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _relation(e, f):
    if set(e) & set(f):
        return "SHARED_ENDPOINT"
    a, b = map(POINTS.__getitem__, e)
    c, d = map(POINTS.__getitem__, f)
    return (
        "PROPER_CROSSING"
        if _orient(a, b, c) * _orient(a, b, d) < 0
        and _orient(c, d, a) * _orient(c, d, b) < 0
        else "DISJOINT"
    )


def _edge(value):
    if (
        not isinstance(value, list)
        or len(value) != 2
        or (not all(type(x) is int and 0 <= x < 5 for x in value))
        or (value[0] >= value[1])
    ):
        raise ValueError
    return tuple(value)


def mathematics(result):
    if not isinstance(result, dict) or set(result) != {
        "selected_edges",
        "pair_classifications",
        "excluded_edge_witnesses",
    }:
        return False
    if not isinstance(result["selected_edges"], list):
        return False
    try:
        selected = [_edge(e) for e in result["selected_edges"]]
    except ValueError:
        return False
    if len(selected) != 5 or selected != sorted(set(selected)):
        return False
    expected_pairs = [
        {"left": list(e), "right": list(f), "relation": _relation(e, f)}
        for e, f in combinations(selected, 2)
    ]
    if any(
        row["relation"] == "DISJOINT" for row in expected_pairs
    ) or not json_value_equal(result["pair_classifications"], expected_pairs):
        return False
    excluded = [e for e in ALL if e not in selected]
    expected = []
    for edge in excluded:
        disjoint = [
            candidate
            for candidate in selected
            if _relation(edge, candidate) == "DISJOINT"
        ]
        if not disjoint:
            return False
        expected.append(
            {"excluded": list(edge), "disjoint_selected": list(disjoint[0])}
        )
    return json_value_equal(result["excluded_edge_witnesses"], expected)


def _write(values):
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(values, sort_keys=True))
    normalize_reward_file(path / "reward.json")


def main():
    submission = load_submission(require_input_binding=False)
    protocol = isinstance(submission, dict)
    result = submission.get("result") if protocol else None
    mathematics_valid = bool(protocol and mathematics(result))
    values = {
        "input_binding": float(workspace_input_is_bound()),
        "protocol": float(protocol),
        "mathematics": float(mathematics_valid),
    }
    reward = float(all(values.values()))
    values.update({"aggregate_reward": reward, "reward": reward})
    _write(values)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _write(
            {
                "protocol": 0.0,
                "input_binding": 0.0,
                "mathematics": 0.0,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
