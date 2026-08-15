from __future__ import annotations

import json
from pathlib import Path

from proof_replay import VerifyResult, verify_submission
from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)


def main() -> None:
    workspace, tests = Path("/app"), Path("/tests")
    frozen_ok = workspace_input_is_bound(
        visible_path=workspace / "input.json", tests=tests
    )
    submission = load_submission(workspace / "submission.json")
    if submission is None:
        submission = {}
    try:
        frozen_input = json.loads((tests / "input.json").read_text())
    except (OSError, ValueError, RecursionError, MemoryError):
        frozen_input = {}
    try:
        result = verify_submission(submission, frozen_input)
    except Exception as exc:  # fail closed at the verifier boundary
        result = VerifyResult(False, f"verifier error: {exc}")
    accepted = bool(frozen_ok and result.correctness)
    reward = 1.0 if accepted else 0.0
    message = "frozen input mismatch" if not frozen_ok else result.message
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(result.correctness),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(logs / "reward.json")
    print(json.dumps({"accepted": accepted, "message": message}, sort_keys=True))


if __name__ == "__main__":
    main()
