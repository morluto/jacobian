"""Fail-closed verifier for an order-664 Hadamard matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

ORDER = 664
HALF = ORDER // 2


def _mathematics(result: Any) -> bool:
    if not isinstance(result, dict) or set(result) != {"rows"}:
        return False
    rows = result["rows"]
    if (
        not isinstance(rows, list)
        or len(rows) != ORDER
        or any(
            not isinstance(row, str)
            or len(row) != ORDER
            or any(bit not in "01" for bit in row)
            for row in rows
        )
        or len(set(rows)) != ORDER
    ):
        return False
    masks = [int(row, 2) for row in rows]
    return all(
        (left ^ right).bit_count() == HALF
        for index, left in enumerate(masks)
        for right in masks[index + 1 :]
    )


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
