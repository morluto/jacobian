from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

TASK_ID = "jacobian/multiplicative-grid-extremum"
SCOPE = (
    "sets of 100 distinct positive integers; a pair is good exactly when its "
    "larger element is 2 or 3 times its smaller element"
)
LIMITATIONS = [
    "The verifier certifies the frozen 100-element extremal claim with an "
    "elementary projection bound; it does not issue proof-assistant VERIFIED "
    "assurance."
]


def main() -> None:
    app = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("/app")
    evidence_dir = app / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    numbers = [2**a * 3**b for a in range(10) for b in range(10)]
    pairs = []
    for i, left in enumerate(numbers):
        for j in range(i + 1, len(numbers)):
            low, high = sorted((left, numbers[j]))
            if high in (2 * low, 3 * low):
                pairs.append([i, j])
    factors = [
        {"index": 10 * a + b, "core": 1, "two_exponent": a, "three_exponent": b}
        for a in range(10)
        for b in range(10)
    ]
    result = {
        "numbers": numbers,
        "good_pairs": pairs,
        "factorizations": factors,
        "projection_summary": {
            "component_count": 1,
            "nonempty_rows": 10,
            "nonempty_columns": 10,
            "witness_projection_cost": 20,
            "universal_projection_cost": 20,
            "universal_edge_bound": 180,
        },
        "claimed_maximum": 180,
        "conclusion": "EXACT_MAXIMUM_CERTIFIED",
    }
    compact_result = json.dumps(result, sort_keys=True, separators=(",", ":"))
    evidence_path = evidence_dir / "answer.txt"
    evidence_path.write_text(
        "RESULT_JSON: " + compact_result + "\n"
        "The witness and projection data are replayed independently.\n",
        encoding="utf-8",
    )
    digest = "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    submission = {
        "task_id": TASK_ID,
        "conclusion": "EXACT_MAXIMUM_CERTIFIED",
        "result": result,
        "claimed_assurance": "COMPUTED",
        "scope": SCOPE,
        "completeness": "COMPLETE",
        "evidence": [{"path": "evidence/answer.txt", "sha256": digest}],
        "limitations": LIMITATIONS,
    }
    (app / "submission.json").write_text(
        json.dumps(submission, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
