from __future__ import annotations

import hashlib
import json
from pathlib import Path

TASK_ID = "jacobian/tutte-flow-domain-audit"
EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 4),
    (5, 7),
    (7, 9),
    (6, 9),
    (6, 8),
    (5, 8),
    (0, 5),
    (1, 6),
    (2, 7),
    (3, 8),
    (4, 9),
]
LIMITATIONS = [
    "ONE_PETERSEN_GRAPH_INSTANCE",
    "MODULAR_FLOW_DOMAIN_AUDIT_ONLY",
    "TUTTE_FIVE_FLOW_CONJECTURE_NOT_ASSESSED",
]


def balances(flow: list[int]) -> list[int]:
    result = [0] * 10
    for value, (source, target) in zip(flow, EDGES, strict=True):
        result[source] = (result[source] + value) % 5
        result[target] = (result[target] - value) % 5
    return result


def main() -> None:
    root = Path("/app")
    flawed = [3, 2, 1, 4, 0, 1, 2, 4, 2, 1, 2, 1, 1, 2, 4]
    repair = [2, 1, 4, 3, 1, 1, 3, 3, 3, 1, 2, 1, 2, 1, 4]
    result = {
        "flawed_flow": flawed,
        "flawed_balances": balances(flawed),
        "zero_edge_index": flawed.index(0),
        "repair_flow": repair,
        "repair_balances": balances(repair),
    }
    payload = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": result,
        "limitations": LIMITATIONS,
    }
    evidence = root / "evidence/answer.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    submission = {
        "task_id": TASK_ID,
        "conclusion": "CONSERVATION_ONLY_IS_UNSOUND_AND_REPAIRED",
        "result": result,
        "claimed_assurance": "CHECKED",
        "scope": "petersen-nowhere-zero-five-flow-audit-v1",
        "completeness": "COMPLETE",
        "evidence": [
            {
                "path": "evidence/answer.json",
                "sha256": "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest(),
            }
        ],
        "limitations": LIMITATIONS,
    }
    (root / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
