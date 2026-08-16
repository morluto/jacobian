"""Produce the Oracle certificate for the finite cuboid scope audit."""

from __future__ import annotations

import json
import sys
from collections import Counter
from math import isqrt
from pathlib import Path

SCOPE = "perfect-cuboid-scope-audit:case-set-v1"
CLASSES = ["PERFECT_CUBOID", "EULER_BRICK_ONLY", "SPACE_AND_TWO_FACES", "OTHER"]


def root(value: int) -> int | None:
    candidate = isqrt(value)
    return candidate if candidate * candidate == value else None


def classify(case: dict[str, object]) -> dict[str, object]:
    edges = case["edges"]
    assert isinstance(edges, list)
    a, b, c = edges
    face_radicands = [a * a + b * b, a * a + c * c, b * b + c * c]
    face_roots = [root(value) for value in face_radicands]
    space_radicand = a * a + b * b + c * c
    space_root = root(space_radicand)
    count = sum(value is not None for value in face_roots)
    if count == 3 and space_root is not None:
        category = "PERFECT_CUBOID"
    elif count == 3:
        category = "EULER_BRICK_ONLY"
    elif count == 2 and space_root is not None:
        category = "SPACE_AND_TWO_FACES"
    else:
        category = "OTHER"
    return {
        "id": case["id"],
        "edges": edges,
        "face_radicands": face_radicands,
        "face_roots": face_roots,
        "space_radicand": space_radicand,
        "space_root": space_root,
        "class": category,
    }


def main() -> None:
    root = (
        Path(sys.argv[2])
        if len(sys.argv) == 3 and sys.argv[1] == "--root"
        else Path("/app")
    )
    frozen = json.loads((root / "input.json").read_text())
    cases = [classify(case) for case in frozen["cases"]]
    counts = Counter(case["class"] for case in cases)
    result = {
        "cases": cases,
        "case_counts": {name: counts[name] for name in CLASSES},
        "contains_perfect_cuboid": counts["PERFECT_CUBOID"] > 0,
        "scope_identity": SCOPE,
    }
    submission = {
        "result": result,
    }
    (root / "submission.json").write_text(
        json.dumps(submission, sort_keys=True, separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
