from __future__ import annotations

import json
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/cycle-double-cover-multiplicity-audit"
SCOPE = "petersen-cycle-double-cover-audit-v1"
LIMITATIONS = [
    "ONE_PETERSEN_GRAPH_INSTANCE",
    "MULTIPLICITY_CONTRACT_AUDIT_ONLY",
    "CYCLE_DOUBLE_COVER_CONJECTURE_NOT_ASSESSED",
]
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
EDGE_INDEX = {tuple(sorted(edge)): i for i, edge in enumerate(EDGES)}


def _canonical(cycle: list[int]) -> tuple[int, ...]:
    variants = []
    for order in (cycle, list(reversed(cycle))):
        variants.extend(tuple(order[i:] + order[:i]) for i in range(len(order)))
    return min(variants)


def _multiplicities(value: object) -> list[int] | None:
    if not isinstance(value, list) or not 4 <= len(value) <= 12:
        return None
    seen: set[tuple[int, ...]] = set()
    counts = [0] * len(EDGES)
    for cycle in value:
        if (
            not isinstance(cycle, list)
            or not 5 <= len(cycle) <= 9
            or len(set(cycle)) != len(cycle)
            or not all(type(v) is int and 0 <= v < 10 for v in cycle)
        ):
            return None
        canonical = _canonical(cycle)
        if canonical in seen:
            return None
        seen.add(canonical)
        for i, left in enumerate(cycle):
            edge = tuple(sorted((left, cycle[(i + 1) % len(cycle)])))
            if edge not in EDGE_INDEX:
                return None
            counts[EDGE_INDEX[edge]] += 1
    return counts


def mathematics(result: object) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "flawed_cycles",
        "flawed_multiplicities",
        "non_double_edge_indices",
        "repair_cycles",
        "repair_multiplicities",
    }:
        return False
    flawed = _multiplicities(result["flawed_cycles"])
    repaired = _multiplicities(result["repair_cycles"])
    if flawed is None or repaired is None:
        return False
    bad = [i for i, count in enumerate(flawed) if count != 2]
    return (
        all(count >= 1 for count in flawed)
        and bool(bad)
        and result["flawed_multiplicities"] == flawed
        and result["non_double_edge_indices"] == bad
        and result["repair_multiplicities"] == repaired
        and repaired == [2] * len(EDGES)
    )


def _raw() -> dict | None:
    try:
        value = json.loads(Path("/app/submission.json").read_text())
    except (OSError, ValueError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _write(values: dict) -> None:
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(values, sort_keys=True))


def main() -> None:
    raw = _raw()
    submission = load_submission(require_input_binding=False)
    contract = strict_submission_contract(
        submission,
        task_id=TASK_ID,
        conclusion="UNION_COVERAGE_IS_INSUFFICIENT_AND_REPAIRED",
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    evidence_ok = bool(
        isinstance(raw, dict)
        and evidence_list_is_bound(raw.get("evidence"), max_bytes=None)
    )
    payload = (
        read_evidence_json(
            raw["evidence"][0], expected_path="evidence/answer.txt", max_bytes=None
        )
        if evidence_ok
        else None
    )
    evidence_ok = bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result", "limitations"}
        and payload.get("schema_version") == "1"
        and payload.get("task_id") == TASK_ID
        and payload.get("result") == raw.get("result")
        and payload.get("limitations") == LIMITATIONS
    )
    values = {
        "input_binding": float(workspace_input_is_bound()),
        "protocol": float(bool(contract)),
        "mathematics": float(
            bool(isinstance(raw, dict) and mathematics(raw.get("result")))
        ),
        "evidence": float(evidence_ok),
        "scope": float(
            bool(
                isinstance(raw, dict)
                and raw.get("scope") == SCOPE
                and raw.get("completeness") == "COMPLETE"
                and raw.get("limitations") == LIMITATIONS
            )
        ),
        "assurance": float(
            bool(
                contract
                and isinstance(raw, dict)
                and raw.get("claimed_assurance")
                in {"UNVERIFIED", "COMPUTED", "CHECKED"}
            )
        ),
    }
    reward = float(all(values.values()))
    values.update(
        {
            "aggregate_reward": reward,
            "reward": reward,
            "false_certification": bool(
                isinstance(raw, dict) and raw.get("claimed_assurance") == "VERIFIED"
            ),
        }
    )
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
                "evidence": 0.0,
                "scope": 0.0,
                "assurance": 0.0,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "false_certification": False,
                "error": type(exc).__name__,
            }
        )
