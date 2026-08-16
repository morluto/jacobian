from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    is_regular_bounded_file,
    json_value_equal,
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

_json_equal = json_value_equal

TASK_ID = "jacobian/illumination-strictness-audit"
VERTICES = list(product((-1, 1), repeat=3))


def _weak(v, d):
    return all((a * b <= 0 for a, b in zip(v, d, strict=True)))


def _strict(v, d):
    return all((a * b < 0 for a, b in zip(v, d, strict=True)))


def _directions(value, n, zeros):
    if not isinstance(value, list) or len(value) != n:
        return False
    if not all(
        isinstance(d, list)
        and len(d) == 3
        and all(type(x) is int and x in {-1, 0, 1} for x in d)
        and (d.count(0) == zeros)
        for d in value
    ):
        return False
    return len({tuple(d) for d in value}) == n


def mathematics(result):
    if not isinstance(result, dict) or set(result) != {
        "flawed_directions",
        "weak_false_positive_pairs",
        "repair_directions",
        "vertex_to_direction",
    }:
        return False
    flawed = result["flawed_directions"]
    repair = result["repair_directions"]
    if not _directions(flawed, 4, 1) or not _directions(repair, 8, 0):
        return False
    weak_cover = all(any(_weak(v, d) for d in flawed) for v in VERTICES)
    expected = [
        {"vertex_index": i, "direction_index": j}
        for i, v in enumerate(VERTICES)
        for j, d in enumerate(flawed)
        if _weak(v, d) and (not _strict(v, d))
    ]
    mapping = result["vertex_to_direction"]
    return (
        weak_cover
        and bool(expected)
        and _json_equal(result["weak_false_positive_pairs"], expected)
        and isinstance(mapping, list)
        and (len(mapping) == 8)
        and all(
            (
                type(j) is int and 0 <= j < 8 and _strict(v, repair[j])
                for v, j in zip(VERTICES, mapping, strict=True)
            )
        )
        and (len(set(mapping)) == 8)
        and all(sum(_strict(v, d) for d in repair) == 1 for v in VERTICES)
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"out-of-range JSON number: {value}")
    return parsed


def _raw(path: Path = Path("/app/submission.json")):
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_finite_json_float,
        )
    except (OSError, ValueError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _write(values):
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    reward_path = path / "reward.json"
    reward_path.write_text(json.dumps(values, sort_keys=True))
    normalize_reward_file(reward_path)


def main():
    submission = load_submission(require_input_binding=False)
    protocol = isinstance(submission, dict)
    mathematics_score = float(bool(protocol and mathematics(submission.get("result"))))
    input_bound = workspace_input_is_bound()
    reward = float(input_bound and protocol and mathematics_score)
    values = {
        "input_binding": float(input_bound),
        "protocol": float(protocol),
        "correctness": mathematics_score,
        "mathematics": mathematics_score,
    }
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
                "correctness": 0.0,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
