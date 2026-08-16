"""Fail-closed verifier for the A(23,6,10) construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)


def _mathematics(result: Any) -> bool:
    if not isinstance(result, dict) or set(result) != {"codewords"}:
        return False
    words = result["codewords"]
    if not isinstance(words, list) or len(words) != 2992:
        return False
    if any(
        not isinstance(word, str)
        or len(word) != 6
        or word[0] not in "01234567"
        or any(digit not in "0123456789abcdef" for digit in word)
        for word in words
    ):
        return False
    if len(set(words)) != len(words):
        return False
    values = [int(word, 16) for word in words]
    if any(value.bit_count() != 10 for value in values):
        return False
    return all(
        (left ^ right).bit_count() >= 6
        for index, left in enumerate(values)
        for right in values[index + 1 :]
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
