from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

TASK_ID = "jacobian/thrackle-local-maximality-certificate"
POINTS = [(0, 0), (4, 0), (5, 3), (2, 5), (-1, 3)]
ALL = list(combinations(range(5), 2))
SELECTED = [(0, 2), (0, 3), (1, 3), (1, 4), (2, 4)]


def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def relation(e, f):
    if set(e) & set(f):
        return "SHARED_ENDPOINT"
    a, b = map(POINTS.__getitem__, e)
    c, d = map(POINTS.__getitem__, f)
    return (
        "PROPER_CROSSING"
        if orient(a, b, c) * orient(a, b, d) < 0
        and orient(c, d, a) * orient(c, d, b) < 0
        else "DISJOINT"
    )


def main():
    root = Path("/app")
    pairs = [
        {"left": list(e), "right": list(f), "relation": relation(e, f)}
        for e, f in combinations(SELECTED, 2)
    ]
    excluded = [e for e in ALL if e not in SELECTED]
    witnesses = [
        {
            "excluded": list(e),
            "disjoint_selected": list(
                next(f for f in SELECTED if relation(e, f) == "DISJOINT")
            ),
        }
        for e in excluded
    ]
    result = {
        "selected_edges": [list(e) for e in SELECTED],
        "pair_classifications": pairs,
        "excluded_edge_witnesses": witnesses,
    }
    payload = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": result,
    }
    evidence = root / "evidence/answer.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    submission = {
        "result": result,
        "witness": [
            {
                "path": "evidence/answer.json",
                "sha256": "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
