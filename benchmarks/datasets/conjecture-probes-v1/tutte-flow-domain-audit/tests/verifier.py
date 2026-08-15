from __future__ import annotations

import json
from pathlib import Path

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    is_regular_bounded_file,
    load_submission,
    read_evidence_json,
    witness_list_is_bound,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/tutte-flow-domain-audit"
SCOPE = "petersen-nowhere-zero-five-flow-audit-v1"
LIMITATIONS = [
    "ONE_PETERSEN_GRAPH_INSTANCE",
    "MODULAR_FLOW_DOMAIN_AUDIT_ONLY",
    "TUTTE_FIVE_FLOW_CONJECTURE_NOT_ASSESSED",
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


def _balances(flow: object) -> list[int] | None:
    if (
        not isinstance(flow, list)
        or len(flow) != 15
        or not all(type(v) is int and 0 <= v < 5 for v in flow)
    ):
        return None
    result = [0] * 10
    for value, (source, target) in zip(flow, EDGES, strict=True):
        result[source] = (result[source] + value) % 5
        result[target] = (result[target] - value) % 5
    return result


def _exact_integer_list(value: object, expected: list[int]) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == len(expected)
        and all(type(item) is int for item in value)
        and value == expected
    )


def _json_equal(left: object, right: object) -> bool:
    pending = [(left, right, 0)]
    visited = 0
    while pending:
        current_left, current_right, depth = pending.pop()
        visited += 1
        if visited > 100_000 or depth > 128:
            return False
        if type(current_left) is not type(current_right):
            return False
        if isinstance(current_left, dict):
            if set(current_left) != set(current_right):
                return False
            pending.extend(
                (current_left[key], current_right[key], depth + 1)
                for key in current_left
            )
        elif isinstance(current_left, list):
            if len(current_left) != len(current_right):
                return False
            pending.extend(
                (a, b, depth + 1)
                for a, b in zip(current_left, current_right, strict=True)
            )
        elif current_left != current_right:
            return False
    return True


def mathematics(result: object) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "flawed_flow",
        "flawed_balances",
        "zero_edge_index",
        "repair_flow",
        "repair_balances",
    }:
        return False
    flawed = _balances(result["flawed_flow"])
    repair = _balances(result["repair_flow"])
    if flawed is None or repair is None:
        return False
    zeros = [i for i, v in enumerate(result["flawed_flow"]) if v == 0]
    return (
        flawed == [0] * 10
        and repair == [0] * 10
        and len(zeros) == 1
        and type(result["zero_edge_index"]) is int
        and result["zero_edge_index"] == zeros[0]
        and _exact_integer_list(result["flawed_balances"], flawed)
        and _exact_integer_list(result["repair_balances"], repair)
        and all(result["repair_flow"])
    )


def _raw():
    path = Path("/app/submission.json")
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, ValueError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _write(values):
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    reward = values.get("reward", 0.0)
    details = {key: value for key, value in values.items() if key != "reward"}
    (path / "reward.json").write_text(json.dumps({"reward": reward}, sort_keys=True))
    (path / "reward-details.json").write_text(json.dumps(details, sort_keys=True))


def _evidence_payload_is_bound(payload: object, result: object) -> bool:
    return bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result"}
        and payload.get("schema_version") == "1"
        and payload.get("task_id") == TASK_ID
        and _json_equal(payload.get("result"), result)
    )


def main():
    raw = _raw()
    submission = load_submission(require_input_binding=False)
    contract = bool(submission)
    evidence_ok = bool(
        isinstance(raw, dict)
        and witness_list_is_bound(raw.get("witness"), max_bytes=None)
    )
    payload = (
        read_evidence_json(
            raw["witness"][0], expected_path="evidence/answer.json", max_bytes=None
        )
        if evidence_ok
        else None
    )
    evidence_ok = _evidence_payload_is_bound(
        payload, raw.get("result") if isinstance(raw, dict) else None
    )
    math_ok = bool(isinstance(raw, dict) and mathematics(raw.get("result")))
    values = {
        "input_binding": float(workspace_input_is_bound()),
        "protocol": float(bool(contract)),
        "mathematics": float(math_ok),
        "correctness": float(math_ok),
        "witness_validity": float(evidence_ok),
    }
    reward = float(all(values.values()))
    values.update(
        {
            "aggregate_reward": reward,
            "reward": reward,
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
                "correctness": 0.0,
                "witness_validity": 0.0,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
