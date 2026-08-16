"""Fail-closed verifier for a projective plane of order 11."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

POINTS = 133
LINES = 133
LINE_SIZE = 12


def _mathematics(result: Any) -> bool:
    if not isinstance(result, dict) or set(result) != {"lines"}:
        return False
    lines = result["lines"]
    if not isinstance(lines, list) or len(lines) != LINES:
        return False
    canonical: list[tuple[int, ...]] = []
    for line in lines:
        if (
            not isinstance(line, list)
            or len(line) != LINE_SIZE
            or any(type(point) is not int or not 0 <= point < POINTS for point in line)
            or len(set(line)) != LINE_SIZE
        ):
            return False
        canonical.append(tuple(sorted(line)))
    if len(set(canonical)) != LINES:
        return False
    pairs = Counter(
        pair for line in canonical for pair in itertools.combinations(line, 2)
    )
    return len(pairs) == POINTS * (POINTS - 1) // 2 and set(pairs.values()) == {1}


def _reward(payload: dict[str, Any]) -> None:
    path = Path("/logs/verifier/reward.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))
    normalize_reward_file(path)


def main() -> None:
    input_binding = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    protocol = submission is not None
    mathematics = bool(protocol and _mathematics(submission.get("result")))
    reward = float(input_binding and protocol and mathematics)
    _reward(
        {
            "input_binding": float(input_binding),
            "protocol_compliance": float(protocol),
            "correctness": float(mathematics),
            "reward": reward,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _reward(
            {
                "input_binding": 0.0,
                "protocol_compliance": 0.0,
                "correctness": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
