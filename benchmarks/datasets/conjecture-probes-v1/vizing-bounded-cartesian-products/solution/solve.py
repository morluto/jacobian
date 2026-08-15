"""Emit the frozen Oracle certificate for the bounded Vizing probe."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

TASK_ID = "jacobian/vizing-bounded-cartesian-products"
LIMITATIONS = [
    "EIGHT_FROZEN_GRAPHS_THIRTEEN_CARTESIAN_PAIRS",
    "NO_GLOBAL_VIZING_CONCLUSION",
]


def main() -> None:
    root = (
        Path(sys.argv[2])
        if len(sys.argv) == 3 and sys.argv[1] == "--root"
        else Path("/app")
    )
    result = json.loads((Path("/solution") / "result.json").read_text())
    evidence = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": result,
        "limitations": LIMITATIONS,
    }
    evidence_path = root / "evidence" / "answer.txt"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n"
    )
    digest = "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    submission = {
        "result": result,
        "witness": [{"path": "evidence/answer.txt", "sha256": digest}],
    }
    (root / "submission.json").write_text(
        json.dumps(submission, sort_keys=True, separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
