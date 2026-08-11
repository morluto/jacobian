from __future__ import annotations

import json
from pathlib import Path

from proof_replay import VerifyResult, verify_submission
from verifier_support import (
    false_verified_claim,
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
        result = verify_submission(workspace, submission, frozen_input)
    except Exception as exc:  # fail closed at the verifier boundary
        result = VerifyResult(
            False, False, False, False, False, f"verifier error: {exc}"
        )
    accepted = bool(
        frozen_ok
        and result.protocol_ok
        and result.correctness
        and result.evidence_validity
        and result.scope_accuracy
        and result.assurance_ok
    )
    reward = 1.0 if accepted else 0.0
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    message = "frozen input mismatch" if not frozen_ok else result.message
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(result.protocol_ok),
                "correctness": float(result.correctness),
                "evidence_validity": float(result.evidence_validity),
                "scope_accuracy": float(result.scope_accuracy),
                "assurance_calibration": float(result.assurance_ok),
                "input_binding": float(frozen_ok),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(logs / "reward.json")
    print(json.dumps({"accepted": accepted, "message": message}, sort_keys=True))


if __name__ == "__main__":
    main()
