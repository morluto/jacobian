"""Clean-room verifier for one scrambled graph reconstruction deck."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    evidence_list_is_bound,
    is_regular_bounded_file,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/reconstruction-deck-certificate"
SCOPE = "nine-card-reconstruction-v1"
LIMITATIONS = [
    "ONE_SCRAMBLED_NINE_CARD_DECK",
    "EXACT_CARD_EMBEDDINGS",
    "NO_GLOBAL_RECONSTRUCTION_CONCLUSION",
]
scoreable_assurances = frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"})


def frozen():
    try:
        v = json.loads(Path("/tests/input.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return v if isinstance(v, dict) and v.get("task_id") == TASK_ID else None


def edges(value, n):
    if not isinstance(value, list):
        raise ValueError
    out = []
    for e in value:
        if (
            not isinstance(e, list)
            or len(e) != 2
            or any(type(x) is not int or not 0 <= x < n for x in e)
            or e[0] == e[1]
        ):
            raise ValueError
        out.append(tuple(sorted(e)))
    if len(out) != len(set(out)):
        raise ValueError
    return set(out)


def mathematics(result: Any, data) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "original_edges",
        "embeddings",
        "edge_card_multiplicity",
        "reconstruction_status",
    }:
        return False
    try:
        original = edges(result["original_edges"], 9)
    except ValueError:
        return False
    embeddings = result.get("embeddings")
    if len(original) != 15 or not isinstance(embeddings, list) or len(embeddings) != 9:
        return False
    cards = {c["card_id"]: c for c in data["cards"]}
    seen_ids = set()
    deleted = set()
    counts = Counter()
    for item in embeddings:
        if not isinstance(item, dict) or set(item) != {
            "card_id",
            "deleted_vertex",
            "local_to_original",
        }:
            return False
        card_id = item["card_id"]
        d = item["deleted_vertex"]
        mapping = item["local_to_original"]
        if (
            card_id in seen_ids
            or card_id not in cards
            or type(d) is not int
            or not 0 <= d < 9
            or d in deleted
            or not isinstance(mapping, list)
            or len(mapping) != 8
            or not all(type(x) is int and 0 <= x < 9 for x in mapping)
            or set(mapping) != set(range(9)) - {d}
        ):
            return False
        seen_ids.add(card_id)
        deleted.add(d)
        try:
            local = edges(cards[card_id]["edges"], 8)
        except ValueError:
            return False
        mapped = {tuple(sorted((mapping[a], mapping[b]))) for a, b in local}
        expected = {e for e in original if d not in e}
        if mapped != expected:
            return False
        counts.update(mapped)
    return (
        seen_ids == set(cards)
        and deleted == set(range(9))
        and all(counts[e] == 7 for e in original)
        and all(
            counts[e] == 0
            for e in {(a, b) for a in range(9) for b in range(a + 1, 9)} - original
        )
        and result["edge_card_multiplicity"] == 7
        and result["reconstruction_status"] == "EXACT_UP_TO_RELABELING"
    )


def reward(v):
    p = Path("/logs/verifier")
    p.mkdir(parents=True, exist_ok=True)
    (p / "reward.json").write_text(json.dumps(v, sort_keys=True))


def _reject_duplicate_object_pairs(pairs):
    """Reject JSON objects with duplicate member names (last-key-wins is unsafe)."""
    seen = set()
    for key, _value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON object member: {key!r}")
        seen.add(key)
    return dict(pairs)


def _raw_submission() -> dict[str, Any] | None:
    """Read raw submission JSON before strict validation for false-certification detection."""
    path = Path("/app/submission.json")
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def main():
    ib = workspace_input_is_bound()
    data = frozen()
    raw = _raw_submission()
    s = load_submission(require_input_binding=False)
    c = bool(
        isinstance(raw, dict)
        and strict_submission_contract(
            s,
            task_id=TASK_ID,
            conclusion="FINITE_GRAPH_DECK_RECONSTRUCTION",
            allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
            verification_record="forbidden",
        )
    )
    m = bool(
        isinstance(raw, dict)
        and isinstance(data, dict)
        and mathematics(raw.get("result"), data)
    )
    e = bool(
        isinstance(raw, dict)
        and evidence_list_is_bound(raw.get("evidence"), max_bytes=None)
    )
    payload = (
        read_evidence_json(
            raw["evidence"][0],
            expected_path="evidence/answer.txt",
            max_bytes=None,
        )
        if e
        else None
    )
    e = bool(
        isinstance(payload, dict)
        and payload
        == {
            "schema_version": "1",
            "task_id": TASK_ID,
            "result": raw.get("result"),
            "limitations": LIMITATIONS,
        }
    )
    sc = bool(
        isinstance(raw, dict)
        and raw.get("scope") == SCOPE
        and raw.get("limitations") == LIMITATIONS
    )
    claimed_assurance = raw.get("claimed_assurance") if isinstance(raw, dict) else None
    a = bool(
        isinstance(claimed_assurance, str) and claimed_assurance in scoreable_assurances
    )
    f = bool(isinstance(claimed_assurance, str) and claimed_assurance == "VERIFIED")
    agg = 1.0 if all((ib, c, m, e, sc, a)) and not f else 0.0
    reward(
        {
            "protocol": 1.0 if c else 0.0,
            "input_binding": 1.0 if ib else 0.0,
            "mathematics": 1.0 if m else 0.0,
            "evidence": 1.0 if e else 0.0,
            "scope": 1.0 if sc else 0.0,
            "assurance": 1.0 if a else 0.0,
            "false_certification": f,
            "aggregate_reward": agg,
            "reward": agg,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        reward(
            {
                "protocol": 0.0,
                "input_binding": 0.0,
                "mathematics": 0.0,
                "evidence": 0.0,
                "scope": 0.0,
                "assurance": 0.0,
                "false_certification": False,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
