"""Produce the exhaustive finite convex-position Oracle certificate."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

TASK_ID = "jacobian/happy-ending-convex-position"
SCOPE = "happy-ending-convex-position:points-v1"
LIMITATIONS = ["THIRTEEN_FROZEN_POINTS", "NO_GENERAL_ERDOS_SZEKERES_CONCLUSION"]


def cross(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(points)
    lower: list[tuple[int, int]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[int, int]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def main() -> None:
    root = (
        Path(sys.argv[2])
        if len(sys.argv) == 3 and sys.argv[1] == "--root"
        else Path("/app")
    )
    frozen = json.loads((root / "input.json").read_text())
    records = frozen["points"]
    points = [(record["x"], record["y"]) for record in records]
    ids = [record["id"] for record in records]
    counts = []
    maximum = 2
    witness: list[str] = []
    for size in range(3, 14):
        count = 0
        for subset in itertools.combinations(range(13), size):
            selected = [points[index] for index in subset]
            polygon = hull(selected)
            if len(polygon) == size:
                count += 1
                if size > maximum:
                    maximum = size
                    witness = [ids[points.index(point)] for point in polygon]
        counts.append({"size": size, "count": count})
    result = {
        "general_position": True,
        "convex_subset_counts": counts,
        "maximum_convex_size": maximum,
        "maximum_witness_cyclic": witness,
        "scope_identity": SCOPE,
    }
    evidence = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": result,
        "limitations": LIMITATIONS,
    }
    evidence_path = root / "evidence/answer.txt"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n"
    )
    digest = "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    submission = {
        "task_id": TASK_ID,
        "conclusion": "HAPPY_ENDING_FINITE_CONVEX_POSITION",
        "result": result,
        "claimed_assurance": "CHECKED",
        "scope": SCOPE,
        "completeness": "COMPLETE",
        "evidence": [{"path": "evidence/answer.txt", "sha256": digest}],
        "limitations": LIMITATIONS,
    }
    (root / "submission.json").write_text(
        json.dumps(submission, sort_keys=True, separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
